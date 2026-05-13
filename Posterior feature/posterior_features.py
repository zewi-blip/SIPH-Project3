import numpy as np
import cv2

class PosteriorFeatureExtractor:

    def __init__(self, posterior_ratio: float = 0.5):
        self.posterior_ratio = posterior_ratio

    # ------------------------------------------------------------------
    # Region helpers
    # ------------------------------------------------------------------
    def _clip(self, val, lo, hi):
        return int(max(lo, min(hi, val)))

    def _get_regions(self, box, image_shape):
        H, W = image_shape[:2]
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        bw     = x1 - x0
        bh     = y1 - y0
        ph     = max(1, int(round(self.posterior_ratio * bh)))
        side_w = max(1, bw // 6)

        post_x0 = self._clip(x0 + side_w,    0, W)
        post_x1 = self._clip(x1 - side_w,    0, W)
        post_y0 = self._clip(y1,             0, H)
        post_y1 = self._clip(y1 + ph,        0, H)

        left_x0  = self._clip(x0 - side_w * 2, 0, W)
        left_x1  = self._clip(x0,              0, W)
        right_x0 = self._clip(x1,              0, W)
        right_x1 = self._clip(x1 + side_w * 2, 0, W)

        return {
            'posterior': (slice(post_y0, post_y1), slice(post_x0, post_x1)),
            'left':      (slice(post_y0, post_y1), slice(left_x0,  left_x1)),
            'right':     (slice(post_y0, post_y1), slice(right_x0, right_x1)),
        }

    # ------------------------------------------------------------------
    # Histogram features
    # ------------------------------------------------------------------
    @staticmethod
    def _histogram_features(patch: np.ndarray) -> dict:
        if patch.size == 0:
            return {k: float('nan')
                    for k in ['mean', 'std', 'skewness', 'energy', 'entropy']}

        gray  = patch.flatten().astype(np.float64)
        counts, _ = np.histogram(gray, bins=256, range=(0, 256))
        p     = counts / (counts.sum() + 1e-10)
        bins  = np.arange(256, dtype=np.float64)

        mean     = float(np.sum(bins * p))
        std      = float(np.sqrt(np.sum((bins - mean) ** 2 * p)))
        skewness = float(np.sum((bins - mean) ** 3 * p))
        energy   = float(np.sum(p ** 2))
        entropy  = float(-np.sum(p * np.log2(p + 1e-10)))

        return {'mean': mean, 'std': std, 'skewness': skewness,
                'energy': energy, 'entropy': entropy}

    # ------------------------------------------------------------------
    # Shadow indicator  (original, unchanged)
    # ------------------------------------------------------------------
    @staticmethod
    def _shadow_indicator(post_mean: float, beside_mean: float) -> int:
        return int(post_mean < beside_mean)

    # ------------------------------------------------------------------
    # Confidence scores  ← only addition
    # ------------------------------------------------------------------
    @staticmethod
    def _confidence_scores(post_mean: float, beside_mean: float) -> tuple:
        eps = 1e-10
        ratio = post_mean / (beside_mean + eps)
        shadow_score      = float(np.clip(1.0 - ratio, 0.0, 1.0))
        enhancement_score = float(np.clip(ratio - 1.0, 0.0, 1.0))
        return shadow_score, enhancement_score

    # ------------------------------------------------------------------
    # Main extraction
    # ------------------------------------------------------------------
    def extract(self, image: np.ndarray, box) -> dict:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        regions     = self._get_regions(box, gray.shape)
        post_feats  = self._histogram_features(gray[regions['posterior']])
        left_feats  = self._histogram_features(gray[regions['left']])
        right_feats = self._histogram_features(gray[regions['right']])

        beside_mean = (left_feats['mean'] + right_feats['mean']) / 2.0

        shadow_score, enhancement_score = self._confidence_scores(
            post_feats['mean'], beside_mean
        )

        features = {}
        for k, v in post_feats.items():
            features[f'posterior_{k}'] = v
        for k, v in left_feats.items():
            features[f'left_{k}'] = v
        for k, v in right_feats.items():
            features[f'right_{k}'] = v

        features.update({
            'beside_mean':       beside_mean,
            'shadow_indicator':  self._shadow_indicator(post_feats['mean'], beside_mean),
            'shadow_score':      shadow_score,       # ← NEW: 0→1 confidence of shadowing
            'enhancement_score': enhancement_score,  # ← NEW: 0→1 confidence of enhancement
        })
        return features