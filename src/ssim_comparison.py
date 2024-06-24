from skimage.metrics import structural_similarity as ssim
import cv2

def compare_ssim(image1, image2):
    # Convert images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Resize images to the same size
    gray_image2 = cv2.resize(gray_image2, (gray_image1.shape[1], gray_image1.shape[0]))

    # Compute SSIM
    score, _ = ssim(gray_image1, gray_image2, full=True)
    return score
