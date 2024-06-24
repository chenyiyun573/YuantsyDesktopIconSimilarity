import cv2

def compare_orb(image1, image2):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect key points and compute descriptors
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Compute similarity score based on matches
    similarity = len(matches)
    return similarity
