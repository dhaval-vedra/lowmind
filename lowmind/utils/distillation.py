"""
LowMind Knowledge Distillation Trainer — optimized for knowledge transfer on edge devices
"""
import numpy as np
from ..core.tensor import Tensor, clip_grad_norm
from ..core.no_grad import no_grad as _no_grad
from .trainer import Trainer


class DistillationTrainer(Trainer):
    """
    Knowledge Distillation Trainer to transfer knowledge from a heavy, pre-trained
    Teacher model to a lightweight Student model.

    Uses temperature-scaled soft loss combining standard hard-label loss and soft-label loss.
    """

    def __init__(self, student_model, teacher_model, optimizer, loss_fn,
                 callbacks=None, clip_grad=0.0, grad_accum_steps=1, verbose=1,
                 temperature=3.0, alpha=0.5):
        super().__init__(
            model=student_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            callbacks=callbacks,
            clip_grad=clip_grad,
            grad_accum_steps=grad_accum_steps,
            verbose=verbose
        )
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha  # weight of soft loss (0.0 to 1.0)
        self.teacher_model.eval()  # Make sure teacher is always in evaluation mode

    def _distillation_loss(self, student_logits, teacher_logits, target):
        """
        Compute standard Knowledge Distillation loss.
        """
        # 1. Hard loss (Standard classification loss)
        hard_loss = self.loss_fn(student_logits, target)

        # 2. Soft loss (KL-divergence or temperature-scaled cross-entropy on soft targets)
        # Scale both student and teacher logits by temperature
        student_scaled = student_logits.data / self.temperature
        teacher_scaled = teacher_logits.data / self.temperature

        # Log-softmax on student scaled logits
        student_max = student_scaled.max(axis=1, keepdims=True)
        student_stable = student_scaled - student_max
        student_log_probs = student_stable - np.log(np.exp(student_stable).sum(axis=1, keepdims=True) + 1e-9)

        # Softmax on teacher scaled logits to get soft targets
        teacher_max = teacher_scaled.max(axis=1, keepdims=True)
        teacher_stable = teacher_scaled - teacher_max
        teacher_probs = np.exp(teacher_stable) / np.exp(teacher_stable).sum(axis=1, keepdims=True)

        # Multi-class KL-divergence soft loss: sum(teacher_prob * log(teacher_prob / student_prob))
        # Or simplified cross-entropy: sum(-teacher_prob * log(student_prob))
        # Since log(student_prob) is student_log_probs:
        soft_loss_val = np.mean(np.sum(-teacher_probs * student_log_probs, axis=1))

        # Scale soft loss by T^2 as mathematically proven in Hinton's paper
        soft_loss_scaled = soft_loss_val * (self.temperature ** 2)

        # Combined loss value
        combined_loss_val = self.alpha * soft_loss_scaled + (1.0 - self.alpha) * hard_loss.item()

        # Wrap in a custom Tensor for backpropagation
        loss = Tensor(np.array([combined_loss_val], dtype=np.float32),
                      requires_grad=student_logits.requires_grad,
                      _children=(student_logits,), _op='distillation_loss')

        def _backward():
            if student_logits.requires_grad:
                student_logits._ensure_grad()

                # Grad of hard loss
                # Calculate hard loss gradient manually
                N = student_logits.data.shape[0]
                shifted = student_logits.data - student_logits.data.max(axis=1, keepdims=True)
                student_probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)

                target_idx = target.data.astype(int).flatten()
                hard_grad = student_probs.copy()
                hard_grad[np.arange(N), target_idx] -= 1
                hard_grad /= N

                # Grad of soft loss
                # Since loss = -T^2 * sum(teacher_probs * log_softmax(student_logits/T))
                # The gradient of KL-div w.r.t student scaled logits is (student_softmax_scaled - teacher_softmax_scaled)
                # d_soft / d_logits = (student_softmax_scaled - teacher_probs) * T^2 * (1/T) / N
                #                    = (student_softmax_scaled - teacher_probs) * T / N
                student_scaled_probs = np.exp(student_stable) / np.exp(student_stable).sum(axis=1, keepdims=True)
                soft_grad = (student_scaled_probs - teacher_probs) * self.temperature / N

                # Combined gradient scaled by incoming loss gradient (respects chain rule & grad accumulation)
                combined_grad = (self.alpha * soft_grad + (1.0 - self.alpha) * hard_grad) * loss.grad
                student_logits.grad += combined_grad

        loss._backward = _backward
        return loss

    def _run_epoch(self, loader, training=True):
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                X, y = batch[0], batch[1]
            else:
                raise ValueError("DataLoader must yield (X, y) tuples.")

            # Get teacher prediction under no_grad to save memory
            with _no_grad():
                teacher_logits = self.teacher_model(X)

            student_logits = self.student_model(X)
            loss = self._distillation_loss(student_logits, teacher_logits, y)

            if self.grad_accum_steps > 1:
                scale_factor = 1.0 / self.grad_accum_steps
                loss_scaled = loss * scale_factor
                loss_scaled.backward()
            else:
                loss.backward()

            total_loss += float(loss.item())
            n_batches += 1

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.clip_grad > 0:
                    clip_grad_norm(self.student_model.parameters(), self.clip_grad)
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Flush any remaining accumulated gradients
        remainder = n_batches % self.grad_accum_steps
        if remainder != 0:
            if self.clip_grad > 0:
                clip_grad_norm(self.student_model.parameters(), self.clip_grad)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss / max(n_batches, 1)
