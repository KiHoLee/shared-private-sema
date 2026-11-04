"""Device-side cost of the scheme against the BERT-base encoder (Comment 5.9, RC5).

Operation count only, no simulation. Writes data/device_cost.txt.

Transmitter per embedding: adapter rotation W x (d^2 MAC) and mask M_u (W x)
(d^2 MAC). Mask rebuild M_u = sum_k A_uk U_k per affinity update: at most U d^2
MAC. Haar bases: U orthogonal d x d matrices drawn once from the shared seed
(QR of a Gaussian matrix, O(d^3) each) and stored.

BERT-base encoder (12 layers, hidden 768, FFN 3072): per layer and token,
Q, K, V and output projections 4 H^2, feed-forward 2 H F; per layer, attention
scores and weighted values 2 n^2 H for n tokens. Embedding lookup, layer norm,
softmax and the mean pooling are not counted, so the encoder count is a lower
bound and the ratio an upper bound.
"""
import os

d, U = 768, 4
L, H, F = 12, 768, 3072


def bert_macs(n):
    return L * (n * (4 * H * H + 2 * H * F) + 2 * n * n * H)


dev_mask = d * d
dev_adapter = d * d
dev_total = dev_mask + dev_adapter
rebuild = U * d * d
store_bytes_f32 = U * d * d * 4

lines = [
    "Device-side cost per embedding at d=%d, U=%d (code/device_cost.py)" % (d, U),
    "mask application d^2 = %d MAC" % dev_mask,
    "adapter rotation d^2 = %d MAC" % dev_adapter,
    "mask plus adapter = %d MAC (%.3g)" % (dev_total, dev_total),
    "mask rebuild per affinity update, at most U d^2 = %d MAC" % rebuild,
    "Haar bases stored, U d^2 float32 = %d bytes = %.2f MB" % (store_bytes_f32, store_bytes_f32 / 1e6),
    "",
    "BERT-base encoder MAC (lower bound) and device share:",
]
for n in (7, 10, 16, 32, 64):
    b = bert_macs(n)
    lines.append("n=%3d tokens  encoder %.4g MAC  mask %.4f%%  mask+adapter %.4f%%"
                 % (n, b, 100.0 * dev_mask / b, 100.0 * dev_total / b))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "device_cost.txt")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
