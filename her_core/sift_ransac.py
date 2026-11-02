"""T5 — SIFT + RANSAC biomedical image analysis for MRI slices."""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

from .preprocess import slice_to_uint8


def _to_bgr(gray_u8: np.ndarray) -> np.ndarray:
    if gray_u8.ndim == 2:
        return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    if gray_u8.ndim == 3 and gray_u8.shape[2] == 3:
        return gray_u8
    raise ValueError(f"Unsupported image shape for visualization: {gray_u8.shape}")


def _encode_png_base64(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode PNG visualization")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _empty_canvas(h: int = 240, w: int = 240, text: str = "No visualization") -> np.ndarray:
    canvas = np.full((h, w, 3), 32, dtype=np.uint8)
    cv2.putText(
        canvas,
        text,
        (20, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return canvas


def analyze_mri_sift_ransac(
    image1,
    image2,
    *,
    ratio_thresh: float = 0.75,
    ransac_reproj_threshold: float = 5.0,
    return_base64: bool = True,
) -> dict[str, Any]:
    """
    Run SIFT feature detection + matching and RANSAC inlier filtering
    on two related MRI slices.

    Parameters
    ----------
    image1, image2:
        MRI slices (2-D arrays). May be float MRI intensities; converted
        to uint8 grayscale using the project's min-max normalization.
    """
    img1 = slice_to_uint8(image1)
    img2 = slice_to_uint8(image2)

    original_1 = _to_bgr(img1)
    original_2 = _to_bgr(img2)

    result: dict[str, Any] = {
        "success": False,
        "message": "",
        "keypoints_image1": 0,
        "keypoints_image2": 0,
        "good_matches": 0,
        "ransac_inliers": 0,
        "ransac_outliers": 0,
        "inlier_ratio": 0.0,
        "homography_found": False,
    }

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    kp1_count = 0 if keypoints1 is None else len(keypoints1)
    kp2_count = 0 if keypoints2 is None else len(keypoints2)
    result["keypoints_image1"] = kp1_count
    result["keypoints_image2"] = kp2_count

    sift_vis_1 = cv2.drawKeypoints(
        original_1,
        keypoints1 or [],
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    sift_vis_2 = cv2.drawKeypoints(
        original_2,
        keypoints2 or [],
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    matches_vis = _empty_canvas(
        max(img1.shape[0], img2.shape[0]),
        img1.shape[1] + img2.shape[1] + 20,
        "Insufficient matches",
    )
    ransac_vis = matches_vis.copy()

    if descriptors1 is None or descriptors2 is None or kp1_count == 0 or kp2_count == 0:
        result["message"] = (
            "No SIFT keypoints/descriptors found in one or both slices. "
            "Try slices with more brain tissue (near the middle of the volume)."
        )
        _attach_images(
            result,
            original_1,
            original_2,
            sift_vis_1,
            sift_vis_2,
            matches_vis,
            ransac_vis,
            return_base64,
        )
        return result

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    result["good_matches"] = len(good)

    matches_vis = cv2.drawMatches(
        img1,
        keypoints1,
        img2,
        keypoints2,
        good,
        None,
        matchColor=(0, 255, 255),
        singlePointColor=(255, 180, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    if len(good) < 4:
        result["message"] = (
            f"Only {len(good)} good SIFT matches found (need at least 4 for RANSAC). "
            "Select two nearby slices from the same modality/patient."
        )
        ransac_vis = cv2.drawMatches(
            img1,
            keypoints1,
            img2,
            keypoints2,
            good,
            None,
            matchColor=(0, 165, 255),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        _attach_images(
            result,
            original_1,
            original_2,
            sift_vis_1,
            sift_vis_2,
            matches_vis,
            ransac_vis,
            return_base64,
        )
        return result

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold)

    if H is None or mask is None:
        result["message"] = (
            "Homography estimation failed. Matches may be too inconsistent for RANSAC."
        )
        ransac_vis = matches_vis
        _attach_images(
            result,
            original_1,
            original_2,
            sift_vis_1,
            sift_vis_2,
            matches_vis,
            ransac_vis,
            return_base64,
        )
        return result

    mask_flat = mask.ravel().astype(int)
    inliers = int(mask_flat.sum())
    outliers = int(len(good) - inliers)
    result["ransac_inliers"] = inliers
    result["ransac_outliers"] = outliers
    result["inlier_ratio"] = float(inliers / len(good)) if good else 0.0
    result["homography_found"] = True
    result["success"] = True
    result["message"] = (
        f"SIFT found {kp1_count} / {kp2_count} keypoints; "
        f"{len(good)} good matches; RANSAC kept {inliers} inliers "
        f"({result['inlier_ratio'] * 100:.1f}%)."
    )

    ransac_vis = cv2.drawMatches(
        img1,
        keypoints1,
        img2,
        keypoints2,
        good,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(80, 80, 80),
        matchesMask=mask_flat.tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    _attach_images(
        result,
        original_1,
        original_2,
        sift_vis_1,
        sift_vis_2,
        matches_vis,
        ransac_vis,
        return_base64,
    )
    return result


def _attach_images(
    result: dict,
    original_1,
    original_2,
    sift_vis_1,
    sift_vis_2,
    matches_vis,
    ransac_vis,
    return_base64: bool,
) -> None:
    images = {
        "original_image1": original_1,
        "original_image2": original_2,
        "sift_visualization_1": sift_vis_1,
        "sift_visualization_2": sift_vis_2,
        "matches_visualization": matches_vis,
        "ransac_visualization": ransac_vis,
    }
    if return_base64:
        for key, img in images.items():
            result[key] = _encode_png_base64(img)
            result[f"{key}_data_url"] = f"data:image/png;base64,{result[key]}"
    else:
        result.update(images)
