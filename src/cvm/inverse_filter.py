import cv2
import numpy as np

def apply_equipotential_filter(image, iter_num=5, kappa=50, gamma=0.1):
    """
    Inverse Problem (Equipotential line method) proxy using Anisotropic Diffusion.
    Preserves edges (cervical bone contours) while smoothing internal noise.
    
    Args:
        image: numpy array (RGB or BGR)
        iter_num: Number of iterations
        kappa: Conduction coefficient (controls edge sensitivity)
        gamma: Integration constant
    
    Returns:
        Filtered image (numpy array)
    """
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        img_gray = image.astype(np.float32)

    img = img_gray.copy()

    for _ in range(iter_num):
        # Calculate differences
        deltaN = np.zeros_like(img)
        deltaS = np.zeros_like(img)
        deltaE = np.zeros_like(img)
        deltaW = np.zeros_like(img)

        deltaN[:-1, :] = np.diff(img, axis=0)
        deltaS[1:, :] = -np.diff(img, axis=0)
        deltaE[:, :-1] = np.diff(img, axis=1)
        deltaW[:, 1:] = -np.diff(img, axis=1)

        # Conduction gradients
        cN = np.exp(-(deltaN/kappa)**2)
        cS = np.exp(-(deltaS/kappa)**2)
        cE = np.exp(-(deltaE/kappa)**2)
        cW = np.exp(-(deltaW/kappa)**2)

        # Update
        img = img + gamma * (cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW)

    img = np.clip(img, 0, 255).astype(np.uint8)
    
    if len(image.shape) == 3:
        # Re-apply to color channels if needed, or just return as merged pseudo-color
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

