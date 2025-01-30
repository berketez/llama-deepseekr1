import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000, scale_base=512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scale_base = scale_base
        
        # YARN için dinamik ölçekleme
        self.scale = (torch.log(torch.tensor(max_seq_len / scale_base)) / 
                     torch.log(torch.tensor(2.0))).floor() + 1
        
        # Frekansları önceden hesapla
        self.precompute_freqs_cis()
        
    def precompute_freqs_cis(self):
        # Dinamik olarak ölçeklenmiş theta değerleri
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        # YARN ölçeklemesi
        theta = theta * self.scale
        
        # Pozisyon vektörü
        seq_idx = torch.arange(self.max_seq_len).float()
        
        # Freqs hesaplama
        freqs = torch.outer(seq_idx, theta)
        
        # Kompleks üstel formda freqs
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        # Yeniden şekillendirme
        freqs_cis = torch.view_as_real(freqs_cis)
        freqs_cis = freqs_cis.view(self.max_seq_len, -1, 2)
        
        self.register_buffer("freqs_cis", freqs_cis)
        
    def apply_rotary_emb(self, x, freqs_cis):
        # x shape: (batch, seq_len, head, dim)
        x_shape = x.shape
        x = x.view(*x_shape[:-1], -1, 2)
        
        # Kompleks çarpım için gerçek ve sanal kısımlar
        x_real, x_imag = x.unbind(-1)
        freqs_cis = freqs_cis[:x.shape[1]]  # Sequence length'e göre kesme
        
        # Kompleks rotasyon
        freqs_real = freqs_cis[..., 0]
        freqs_imag = freqs_cis[..., 1]
        
        x_out_real = x_real * freqs_real - x_imag * freqs_imag
        x_out_imag = x_real * freqs_imag + x_imag * freqs_real
        
        # Sonucu birleştir
        x_out = torch.stack([x_out_real, x_out_imag], dim=-1)
        
        # Orijinal boyuta geri dön
        x_out = x_out.view(*x_shape)
        
        return x_out

    def forward(self, x, seq_len):
        return self.freqs_cis[:seq_len]

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)

    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, freqs_cis):
        # Yeni YARN-based rotary embedding uygulama
        q_rot = self.rotary_emb.apply_rotary_emb(q, freqs_cis)
        k_rot = self.rotary_emb.apply_rotary_emb(k, freqs_cis)
        return q_rot, k_rot

    def forward(self, x, mask=None):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # YARN-based rotary embeddings uygulama
        freqs_cis = self.rotary_emb(x, L)
        q, k = self.apply_rotary_pos_emb(q, k, freqs_cis)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_dropout=attn_dropout)
        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x

class Llama(nn.Module):
    def __init__(
        self,
        vocab_size=32000,
        dim=4096,
        depth=32,
        num_heads=32,
        mlp_ratio=4.0,
        dropout=0.0,
        attn_dropout=0.0,
        max_seq_len=2048,
        cot_tokens=5,
        thinking_dropout=0.0,  # Düşünme modu için özel dropout
        min_reflection_steps=1,
        max_reflection_steps=5
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        
        # CoT için özel tokenler
        self.think_token = vocab_size
        self.end_think_token = vocab_size + 1
        self.answer_token = vocab_size + 2
        self.end_answer_token = vocab_size + 3
        
        # Embedding ve çıktı katmanını genişletiyoruz
        self.token_emb = nn.Embedding(vocab_size + cot_tokens, dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=False,
                dropout=dropout,
                attn_dropout=attn_dropout
            )
            for _ in range(depth)
        ])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size + cot_tokens, bias=False)
        
        # Düşünme modu parametreleri
        self.thinking_dropout = thinking_dropout
        self.base_attn_dropout = attn_dropout
        self.min_reflection_steps = min_reflection_steps
        self.max_reflection_steps = max_reflection_steps
        self.thinking_depth = 0
        self.reflection_threshold = 0.7
        
        # Düşünme modu durumu
        self.thinking_state = {
            'active': False,
            'depth': 0,
            'confidence': 0.0,
            'reflection_count': 0,
            'last_reward': 0.0
        }
        
    def adjust_thinking_params(self, reward=None):
        """Düşünme parametrelerini RL geri bildirimine göre ayarla"""
        if reward is not None:
            self.thinking_state['last_reward'] = reward
            
            # Ödüle göre düşünme derinliğini ayarla
            if reward < 0.3:
                self.thinking_depth = min(self.thinking_depth + 1, self.max_reflection_steps)
            elif reward > 0.7:
                self.thinking_depth = max(self.thinking_depth - 1, self.min_reflection_steps)
                
            # Confidence skorunu güncelle
            self.thinking_state['confidence'] = (
                0.9 * self.thinking_state['confidence'] + 0.1 * reward
            )
    
    def start_thinking(self, intensity=1.0):
        """Düşünme modunu başlat ve parametreleri ayarla"""
        self.thinking_state['active'] = True
        self.thinking_state['depth'] = 0
        
        # Düşünme yoğunluğuna göre dropout ayarla
        for layer in self.layers:
            if hasattr(layer.attn, 'attn_dropout'):
                layer.attn.attn_dropout.p = self.thinking_dropout * intensity
    
    def end_thinking(self):
        """Düşünme modunu sonlandır ve normal parametrelere dön"""
        self.thinking_state['active'] = False
        
        # Normal dropout değerlerine geri dön
        for layer in self.layers:
            if hasattr(layer.attn, 'attn_dropout'):
                layer.attn.attn_dropout.p = self.base_attn_dropout
    
    def forward(self, x, mask=None):
        x = self.token_emb(x)
        
        # Düşünme modunda ekstra işlemler
        if self.thinking_state['active']:
            # Düşünme maskesi - daha fazla context görünürlüğü
            think_mask = torch.ones_like(mask) if mask is not None else None
            
            # Düşünme derinliğine göre tekrarlı işlem
            for _ in range(self.thinking_depth):
                for layer in self.layers:
                    x = layer(x, think_mask)
                self.thinking_state['depth'] += 1
                
                # Confidence kontrolü
                if self.thinking_state['confidence'] > self.reflection_threshold:
                    break
        else:
            # Normal forward pass
            for layer in self.layers:
                x = layer(x, mask)
        
        x = self.norm(x)
        logits = self.head(x)
        
        # Düşünme modunda token vurgulaması
        if self.thinking_state['active']:
            # Düşünme derinliğine göre vurgulama şiddetini ayarla
            emphasis = 1.0 + (self.thinking_state['depth'] / self.max_reflection_steps)
            logits[..., self.think_token:self.end_answer_token+1] *= emphasis
            
        return logits

    def generate_with_cot(self, input_ids, max_length=100, thinking_intensity=1.0):
        """Chain-of-Thought ile geliştirilmiş çıktı üretimi"""
        self.start_thinking(intensity=thinking_intensity)
        
        # Düşünme aşaması
        think_output = self.generate(
            input_ids=torch.cat([input_ids, torch.tensor([self.think_token])]),
            max_length=max_length
        )
        
        # Düşünme kalitesini değerlendir
        if hasattr(self, 'rl_module'):
            reflection_quality = self.rl_module.verification_score(think_output).mean()
            self.adjust_thinking_params(reflection_quality)
        
        self.end_thinking()
        
        # Cevap aşaması
        final_output = self.generate(
            input_ids=torch.cat([think_output, torch.tensor([self.answer_token])]),
            max_length=max_length
        )
        
        return final_output

class RLModule(nn.Module):
    def __init__(self, model_dim, num_groups=4):
        super().__init__()
        self.model_dim = model_dim
        self.num_groups = num_groups
        
        # GRPO için grup başına değer tahminleyicileri
        self.group_value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, model_dim // 2),
                nn.ReLU(),
                nn.Linear(model_dim // 2, 1)
            ) for _ in range(num_groups)
        ])
        
        # Reflection ve verification başlıkları
        self.reflection_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )
        
        self.verification_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Linear(model_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Dil tespiti için özel katman
        self.language_detector = nn.Sequential(
            nn.Linear(model_dim, 512),
            nn.ReLU(),
            nn.Linear(512, len(SUPPORTED_LANGUAGES))  # Desteklenen diller için çıktı
        )
        
        # Format kontrolü için
        self.format_checker = nn.Sequential(
            nn.Linear(model_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # 4 format özelliği: think başlangıç/bitiş, answer başlangıç/bitiş
        )
        
        # GRPO için grup ağırlıkları
        self.group_weights = nn.Parameter(torch.ones(num_groups))
        
    def compute_group_values(self, hidden_states):
        """GRPO için grup değerlerini hesapla"""
        group_values = []
        batch_size = hidden_states.shape[0]
        
        # Her grup için değer hesapla
        for i in range(self.num_groups):
            start_idx = i * (batch_size // self.num_groups)
            end_idx = (i + 1) * (batch_size // self.num_groups)
            group_hidden = hidden_states[start_idx:end_idx]
            group_value = self.group_value_heads[i](group_hidden)
            group_values.append(group_value)
            
        return torch.cat(group_values, dim=0)
    
    def compute_accuracy_reward(self, output, target, temperature=1.0):
        """Doğruluk ödülü hesaplama"""
        if target is None:
            return 0.0
            
        # Cosine benzerliği
        similarity = F.cosine_similarity(output, target, dim=-1)
        
        # Sıcaklık ile ölçekleme
        scaled_similarity = similarity / temperature
        
        # Pozitif örnekleri vurgula
        reward = torch.where(
            scaled_similarity > 0.8,
            scaled_similarity * 1.2,  # Bonus
            scaled_similarity
        )
        
        return reward.mean()
    
    def compute_format_reward(self, hidden_states, output_ids):
        """Format ödülü hesaplama"""
        # Format özelliklerini tahmin et
        format_logits = self.format_checker(hidden_states)
        format_probs = torch.sigmoid(format_logits)
        
        # Beklenen format yapısı
        expected_format = {
            'think_start': 0,
            'think_end': 1,
            'answer_start': 2,
            'answer_end': 3
        }
        
        format_score = 0.0
        last_position = -1
        
        # Format sıralamasını kontrol et
        for token_id in output_ids:
            current_pos = -1
            
            if token_id == self.think_token:
                current_pos = expected_format['think_start']
            elif token_id == self.end_think_token:
                current_pos = expected_format['think_end']
            elif token_id == self.answer_token:
                current_pos = expected_format['answer_start']
            elif token_id == self.end_answer_token:
                current_pos = expected_format['answer_end']
                
            if current_pos != -1:
                if current_pos > last_position:
                    format_score += 0.25
                    format_score += format_probs[:, current_pos].mean()
                else:
                    format_score -= 0.5  # Yanlış sıralama cezası
                last_position = current_pos
                
        return torch.tensor(format_score, device=hidden_states.device)
    
    def compute_language_consistency_reward(self, hidden_states, input_language):
        """Dil tutarlılığı ödülü hesaplama"""
        if input_language is None:
            return 0.0
            
        # Dil tespiti
        language_logits = self.language_detector(hidden_states)
        language_probs = F.softmax(language_logits, dim=-1)
        
        # Hedef dil indeksi
        target_lang_idx = SUPPORTED_LANGUAGES.index(input_language)
        
        # Dil tutarlılığı skoru
        language_consistency = language_probs[:, target_lang_idx].mean()
        
        # Karışık dil cezası
        entropy = -(language_probs * torch.log(language_probs + 1e-10)).sum(dim=-1).mean()
        
        return language_consistency - 0.1 * entropy
    
    def forward(self, hidden_states, output_ids=None, target=None, input_language=None):
        """Ana forward geçişi"""
        # GRPO değerleri
        group_values = self.compute_group_values(hidden_states)
        
        # Ödül hesaplamaları
        rewards = {
            'accuracy': self.compute_accuracy_reward(hidden_states, target),
            'format': self.compute_format_reward(hidden_states, output_ids) if output_ids is not None else 0.0,
            'language': self.compute_language_consistency_reward(hidden_states, input_language)
        }
        
        # Reflection ve verification
        reflection_states = self.reflection_head(hidden_states)
        verification_score = self.verification_head(hidden_states)
        
        # Toplam ödül
        total_reward = (
            0.4 * rewards['accuracy'] +
            0.3 * rewards['format'] +
            0.3 * rewards['language']
        )
        
        return {
            'group_values': group_values,
            'rewards': rewards,
            'reflection_states': reflection_states,
            'verification_score': verification_score,
            'total_reward': total_reward
        }

# Desteklenen diller
SUPPORTED_LANGUAGES = ['en', 'tr', 'de', 'fr', 'es', 'it']

class SelfImprovingLlama(Llama):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rl_module = RLModule(self.dim, num_groups=4)
        
        # GRPO parametreleri
        self.clip_ratio = 0.2  # PPO clip parametresi
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        
        # Yansıtma parametreleri
        self.min_reflection_depth = 1
        self.max_reflection_depth = 5
        self.reflection_temperature = 1.0
        self.reflection_decay = 0.95
        self.reflection_success_threshold = 0.75
        self.reflection_history = []
        
    def self_reflect(self, hidden_states, output_logits, step_count=0):
        """Gelişmiş yansıtma mekanizması"""
        # Mevcut durumun kalitesini değerlendir
        reflection_score = self.rl_module.verification_head(hidden_states).sigmoid()
        current_quality = reflection_score.mean()
        
        # Yansıtma geçmişini kontrol et
        if len(self.reflection_history) > 0:
            avg_historical_quality = sum(self.reflection_history[-5:]) / min(5, len(self.reflection_history))
            needs_deeper_reflection = current_quality < avg_historical_quality
        else:
            needs_deeper_reflection = current_quality < self.reflection_threshold
        
        # Dinamik yansıtma derinliği hesaplama
        if needs_deeper_reflection:
            # Düşük kalite durumunda daha derin yansıtma
            target_depth = min(
                self.max_reflection_depth,
                self.thinking_depth + 1 + int((1 - current_quality) * 3)
            )
        else:
            # İyi kalite durumunda daha sığ yansıtma
            target_depth = max(
                self.min_reflection_depth,
                self.thinking_depth - 1
            )
        
        # Erken durma kontrolü
        if step_count >= target_depth or current_quality > self.reflection_success_threshold:
            self.reflection_history.append(current_quality.item())
            return output_logits, current_quality
        
        # Yansıtma işlemi
        reflection_states = self.rl_module.reflection_head(hidden_states)
        
        # Multi-head yansıtma
        reflection_heads = []
        for _ in range(min(4, target_depth - step_count)):
            # Farklı perspektiflerden yansıtma
            head_output = self.forward(reflection_states)
            reflection_heads.append(head_output)
            
            # Her head'in kalitesini değerlendir
            head_quality = self.rl_module.verification_head(head_output).sigmoid()
            if head_quality.mean() > current_quality:
                reflection_states = head_output
                current_quality = head_quality.mean()
        
        # En iyi head'i seç
        if reflection_heads:
            head_qualities = torch.stack([
                self.rl_module.verification_head(head).sigmoid().mean()
                for head in reflection_heads
            ])
            
            # Sıcaklık ile yumuşatılmış seçim
            weights = F.softmax(head_qualities / self.reflection_temperature, dim=0)
            
            # Ağırlıklı kombinasyon
            refined_output = sum(
                head * weight 
                for head, weight in zip(reflection_heads, weights)
            )
            
            # Sıcaklığı güncelle
            self.reflection_temperature *= self.reflection_decay
        else:
            refined_output = reflection_states
        
        # Yansıtma sonuçlarını kaydet
        self.reflection_history.append(current_quality.item())
        
        # Rekürsif yansıtma
        return self.self_reflect(
            refined_output,
            output_logits,
            step_count + 1
        )
    
    def adjust_reflection_params(self, reward):
        """Yansıtma parametrelerini ödüle göre ayarla"""
        # Başarı eşiğini dinamik olarak ayarla
        if reward > self.reflection_success_threshold:
            self.reflection_success_threshold = min(
                0.9,
                self.reflection_success_threshold * 1.05
            )
        else:
            self.reflection_success_threshold = max(
                0.6,
                self.reflection_success_threshold * 0.95
            )
            
        # Sıcaklık parametresini ayarla
        if reward < 0.3:
            self.reflection_temperature = min(2.0, self.reflection_temperature * 1.1)
        elif reward > 0.7:
            self.reflection_temperature = max(0.5, self.reflection_temperature * 0.9)

    def compute_policy_loss(self, old_values, new_values, advantages, rewards, entropy_coef=0.01, value_coef=0.5):
        """GRPO ve değer loss'larını birleştiren gelişmiş kayıp hesaplama"""
        
        # Policy ratio hesaplama
        ratio = torch.exp(new_values - old_values.detach())
        
        # GRPO clipping
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        
        # Policy loss bileşenleri
        policy_loss_1 = ratio * advantages
        policy_loss_2 = clipped_ratio * advantages
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        # Değer loss'u (TD error)
        value_pred = new_values
        value_targets = rewards
        value_loss = F.mse_loss(value_pred, value_targets)
        
        # Entropi bonusu
        probs = F.softmax(new_values, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        
        # Toplam loss
        total_loss = (
            policy_loss + 
            value_coef * value_loss - 
            entropy_coef * entropy
        )
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy
        }
        
    def compute_reward(self, output_states, output_ids, target=None, input_language=None):
        """Gelişmiş ödül hesaplama"""
        # RL modülünden ödülleri al
        rl_outputs = self.rl_module(
            hidden_states=output_states,
            output_ids=output_ids,
            target=target,
            input_language=input_language
        )
        
        # Temel ödüller
        rewards = rl_outputs['rewards']
        
        # Yansıtma kalitesi bonusu
        reflection_quality = rl_outputs['verification_score'].mean()
        
        # Düşünme derinliği cezası/bonusu
        depth_factor = torch.tensor(
            1.0 - (self.thinking_state['depth'] / self.max_reflection_steps)
            if self.thinking_state['depth'] > 0 else 1.0
        )
        
        # Uzun vadeli ödül hesaplama
        if len(self.reflection_history) > 0:
            historical_avg = sum(self.reflection_history[-5:]) / min(5, len(self.reflection_history))
            improvement_bonus = max(0, (rewards['accuracy'] - historical_avg) * 0.2)
        else:
            improvement_bonus = 0.0
            
        # Toplam ödül hesaplama
        total_reward = (
            0.4 * rewards['accuracy'] +
            0.3 * rewards['format'] +
            0.2 * rewards['language'] +
            0.1 * reflection_quality
        ) * depth_factor + improvement_bonus
        
        # Ödül geçmişini güncelle
        self.reflection_history.append(total_reward.item())
        
        # GRPO için grup değerlerini döndür
        return {
            'total_reward': total_reward,
            'group_values': rl_outputs['group_values'],
            'reflection_states': rl_outputs['reflection_states'],
            'components': rewards
        }
        
    def train_step(self, batch, optimizer):
        """Tek eğitim adımı"""
        # Forward pass
        outputs = self(batch['input_ids'], batch.get('attention_mask'))
        
        # Ödül hesaplama
        reward_info = self.compute_reward(
            output_states=outputs,
            output_ids=batch['output_ids'],
            target=batch.get('target'),
            input_language=batch.get('language')
        )
        
        # GRPO kaybı hesaplama
        old_values = reward_info['group_values'].detach()
        new_values = self.rl_module.compute_group_values(outputs)
        advantages = reward_info['total_reward'] - old_values
        
        loss_info = self.compute_policy_loss(
            old_values=old_values,
            new_values=new_values,
            advantages=advantages,
            rewards=reward_info['total_reward']
        )
        
        # Optimizasyon
        optimizer.zero_grad()
        loss_info['total_loss'].backward()
        optimizer.step()
        
        return {
            'loss': loss_info['total_loss'].item(),
            'reward': reward_info['total_reward'].mean().item(),
            'accuracy': reward_info['components']['accuracy'].mean().item(),
            'format_score': reward_info['components']['format'].mean().item(),
            'language_score': reward_info['components']['language'].mean().item()
        }

    def forward_with_rl(self, x, mask=None, target=None, input_language=None):
        """RL geri bildirimiyle geliştirilmiş forward geçişi"""
        batch_size = x.shape[0]
        
        # İlk forward geçişi
        initial_hidden_states = self.token_emb(x)
        
        # GRPO için grup değerlerini hesapla
        initial_values = self.rl_module.compute_group_values(initial_hidden_states)
        
        # Düşünme modunda ekstra işlemler
        if self.thinking_state['active']:
            think_mask = torch.ones_like(mask) if mask is not None else None
            
            hidden_states = initial_hidden_states
            accumulated_rewards = []
            
            # Her düşünme adımı için
            for step in range(self.thinking_depth):
                # Layer geçişleri
                for layer in self.layers:
                    hidden_states = layer(hidden_states, think_mask)
                
                # Ara ödül hesaplama
                step_reward = self.rl_module(
                    hidden_states=hidden_states,
                    target=target,
                    input_language=input_language
                )
                accumulated_rewards.append(step_reward['total_reward'])
                
                # Erken durma kontrolü
                if step_reward['verification_score'].mean() > self.reflection_threshold:
                    break
                    
            # Son hidden states
            final_hidden_states = hidden_states
            
            # Ortalama ödül hesaplama
            mean_reward = torch.stack(accumulated_rewards).mean()
            
            # Düşünme parametrelerini güncelle
            self.adjust_thinking_params(mean_reward)
            
        else:
            # Normal forward geçişi
            final_hidden_states = initial_hidden_states
            for layer in self.layers:
                final_hidden_states = layer(final_hidden_states, mask)
        
        # Son norm ve logits
        final_hidden_states = self.norm(final_hidden_states)
        logits = self.head(final_hidden_states)
        
        # GRPO için son değerleri hesapla
        final_values = self.rl_module.compute_group_values(final_hidden_states)
        
        # Avantaj hesaplama
        rewards = self.compute_reward(
            final_hidden_states,
            None,  # output_ids henüz yok
            target,
            input_language
        )
        advantages = rewards['total_reward'] - initial_values
        
        # GRPO kaybını hesapla
        loss_info = self.compute_policy_loss(
            old_values=initial_values,
            new_values=final_values,
            advantages=advantages,
            rewards=rewards['total_reward']
        )
        
        return {
            'logits': logits,
            'hidden_states': final_hidden_states,
            'policy_loss': loss_info['policy_loss'],
            'rewards': rewards,
            'values': final_values
        }

    def generate_with_improvement(
        self, 
        input_ids, 
        max_length=100,
        num_iterations=3,
        improvement_threshold=0.1,
        temperature=1.0,
        top_k=50,
        top_p=0.9
    ):
        """İteratif RL iyileştirmesi ile geliştirilmiş çıktı üretimi"""
        
        best_output = None
        best_reward = float('-inf')
        improvement_history = []
        
        # İteratif iyileştirme döngüsü
        for iteration in range(num_iterations):
            # Düşünme modunu aktifleştir
            self.start_thinking(intensity=1.0 - (iteration / num_iterations))
            
            # Forward geçişi ile çıktı üret
            outputs = self.forward_with_rl(
                x=input_ids,
                mask=None,  # İlk aşamada mask yok
                target=best_output if best_output is not None else None
            )
            
            # Sampling parametrelerini ayarla
            logits = outputs['logits']
            if temperature != 1.0:
                logits = logits / temperature
                
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
                
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Olasılık dağılımını hesapla
            probs = F.softmax(logits, dim=-1)
            
            # Çıktı tokenlerini örnekle
            current_output = torch.multinomial(probs[0], 1)
            current_length = 1

            # max_length'e ulaşana kadar token üretmeye devam et
            while current_length < max_length:
                next_token = torch.multinomial(probs[current_length-1], 1)
                current_output = torch.cat([current_output, next_token], dim=0)
                current_length += 1
                
                # Eğer EOS token gelirse dur
                if next_token.item() == self.end_answer_token:
                    break
            
            # Alternatif çözümleri değerlendir
            alternative_rewards = []
            for _ in range(3):  # 3 farklı alternatif dene
                alt_output = torch.multinomial(probs[0], 1)
                alt_outputs = self.forward_with_rl(
                    x=input_ids,
                    mask=None,
                    target=alt_output
                )
                alternative_rewards.append(alt_outputs['rewards']['total_reward'])
            
            # En iyi alternatifi seç
            current_reward = outputs['rewards']['total_reward']
            max_alt_reward = max(alternative_rewards)
            
            if max_alt_reward > current_reward:
                current_reward = max_alt_reward
                current_output = alternative_rewards.index(max_alt_reward)
            
            # İyileştirme geçmişini güncelle
            improvement = (current_reward - best_reward) if best_output is not None else current_reward
            improvement_history.append(improvement)
            
            # En iyi sonucu güncelle
            if current_reward > best_reward:
                best_reward = current_reward
                best_output = current_output
                
                # Yansıtma ve öğrenme
                self.rl_module.adjust_reflection_params(current_reward)
                
                # Düşünme parametrelerini güncelle
                self.adjust_thinking_params(current_reward)
            
            # Erken durma kontrolü
            if len(improvement_history) >= 2:
                recent_improvement = improvement_history[-1] - improvement_history[-2]
                if recent_improvement < improvement_threshold:
                    break
            
            # Düşünme modunu kapat
            self.end_thinking()
        
        # Son çıktıyı hazırla
        final_output = {
            'output_ids': best_output,
            'reward': best_reward,
            'improvement_history': improvement_history,
            'num_iterations': iteration + 1
        }
        
        # Eğer CoT kullanılıyorsa, düşünme sürecini de ekle
        if self.thinking_state['active']:
            final_output['thinking_process'] = {
                'depth': self.thinking_state['depth'],
                'confidence': self.thinking_state['confidence'],
                'reflection_count': self.thinking_state['reflection_count']
            }
        
        return final_output

    def evaluate_alternative(self, input_ids, output, temperature=0.7):
        """Alternatif çözümleri değerlendir"""
        # Mevcut çözümün reward'ını hesapla
        current_outputs = self.forward_with_rl(
            x=input_ids,
            target=output
        )
        current_reward = current_outputs['rewards']['total_reward']
        
        # Alternatif çözümler üret ve değerlendir
        alternatives = []
        for temp in [temperature * 0.8, temperature, temperature * 1.2]:
            alt_output = self.generate(
                input_ids=input_ids,
                max_length=output.size(1),
                temperature=temp
            )
            
            alt_outputs = self.forward_with_rl(
                x=input_ids,
                target=alt_output
            )
            
            alternatives.append({
                'output': alt_output,
                'reward': alt_outputs['rewards']['total_reward'],
                'temperature': temp
            })
        
        # En iyi alternatifi seç
        best_alt = max(alternatives, key=lambda x: x['reward'])
        
        return {
            'current_reward': current_reward,
            'best_alternative': best_alt,
            'all_alternatives': alternatives
        }
