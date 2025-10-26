import torch
import torch.nn as nn


class FeedbackLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, feedback):
        # Save input and weights for backward pass
        ctx.save_for_backward(input)
        ctx.feedback = feedback
        output = input @ weight.T + bias  # Standard forward computation
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, = ctx.saved_tensors
        feedback = ctx.feedback

        # Compute gradients with feedback alignment
        grad_input = grad_output @ feedback.T  # Apply feedback weights
        grad_weight = grad_output.T @ input
        grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None  # None for feedback (fixed)


class FeedbackLinear(nn.Module):
    def __init__(self, in_features, out_features, feedback=None):
        super(FeedbackLinear, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

        self.register_buffer('feedback',
                             feedback if feedback is not None else torch.randn(in_features, out_features))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return FeedbackLinearFunction.apply(x, self.weight, self.bias, self.feedback)

    def to(self, *args, **kwargs):
        super(FeedbackLinear, self).to(*args, **kwargs)
        self.weight.data = self.weight.data.to(*args, **kwargs)
        self.bias.data = self.bias.data.to(*args, **kwargs)
        return self
