import cv2
from src.histogram_comparison import compare_histograms
from src.ssim_comparison import compare_ssim
from src.feature_matching import compare_orb
from src.deep_learning_comparison import compare_deep_learning

def main():
    # Load images
    image1 = cv2.imread('images/image1.png')
    image2 = cv2.imread('images/image2.png')

    # Compare images using different methods
    hist_similarity = compare_histograms(image1, image2)
    ssim_similarity = compare_ssim(image1, image2)
    orb_similarity = compare_orb(image1, image2)
    # dl_similarity = compare_deep_learning(image1, image2)

    # Print results
    print(f'Histogram Similarity: {hist_similarity}')
    print(f'SSIM Similarity: {ssim_similarity}')
    print(f'ORB Feature Matching Similarity: {orb_similarity}')
    # print(f'Deep Learning Feature Similarity: {dl_similarity}')

if __name__ == '__main__':
    main()
