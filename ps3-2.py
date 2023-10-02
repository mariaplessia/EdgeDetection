import numpy as np
import cv2
import os
import argparse


def canny(img):

    # Denoise image and convert to greyscale to prepare for canny
    denoised = cv2.GaussianBlur(img, (5,5), 0)
    grayscale = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    # Initialize thresholds and aperture
    threshold1 = 50 
    threshold2 = 150 
    aperture = 3
    
    # Set up slidebars for low and high thresholds and aperture
    window_name1 = 'Canny Edge Detector'

    cv2.namedWindow(window_name1, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('threshold1', window_name1, threshold1, 255, lambda x: None)
    cv2.createTrackbar('threshold2', window_name1, threshold2, 255, lambda x: None)
    cv2.createTrackbar('aperture', window_name1, aperture, 7, lambda x: None)

    canny = cv2.Canny(grayscale, threshold1, threshold2, apertureSize = aperture, L2gradient = True)
    invert = cv2.bitwise_not(canny)

    # Loop through live slidebar changes
    while True:
        # Update parameters as the slidebars move
        threshold1 = cv2.getTrackbarPos('threshold1', window_name1)
        threshold2 = cv2.getTrackbarPos('threshold2', window_name1)
        aperture = cv2.getTrackbarPos('aperture', window_name1)
        
        # Update canny filter with the updated parameters
        canny = cv2.Canny(grayscale, threshold1, threshold2, aperture, L2gradient=True)

        # Change image to white background and black lines
        invert = cv2.bitwise_not(canny)  
        cv2.imshow(window_name1, invert)
        
        # Exit loop if 'esc' is pressed
        if cv2.waitKey(1) == 27:
            break

    # Save final image
    cv2.imwrite(os.path.splitext(args.input)[0]+"-canny.png", invert)
    

def sobel(img):

    # Convert to greyscale to prepare for sobel filter
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Horizontal Sobel Filter
    # Horizontal Derivative
    hd_kernel = 0.5*np.array([[0,0,0],
                            [-1,0,1],
                            [0,0,0]])
    horizontal_derivative = cv2.filter2D(grayscale, -1, hd_kernel)
    # Vertical Gaussian Filter
    vertical_gaussian = 0.25*np.array([[0,1,0],
                                    [0,2,0],
                                    [0,1,0]])
    horizontal_sobel = cv2.filter2D(horizontal_derivative, -1, vertical_gaussian)


    # Vertical Sobel Filter
    # Vertical Derivative
    vd_kernel = 0.5*np.array([[0,1,0],
                            [0,0,0],
                            [0,-1,0]])
    vertical_derivative = cv2.filter2D(grayscale, -1, vd_kernel)
    # Horizontal Gaussian Filter
    horizontal_gaussian = 0.25*np.array([[0,0,0],
                                        [1,2,1],
                                        [0,0,0]])
    vertical_sobel = cv2.filter2D(vertical_derivative, -1, horizontal_gaussian)

    sobel_final = horizontal_sobel + vertical_sobel
    invert = cv2.bitwise_not(sobel_final) 

    # Save & display final image
    cv2.imwrite(os.path.splitext(args.input)[0]+"-sobel.png", invert)
    window_name2 = 'Custom Soble Edge Detector'
    cv2.imshow(window_name2, invert)
    cv2.waitKey(5000)


def main(args):
    
    dir = "ps3-images/"
    original = cv2.imread(dir+args.input)

    # Display the input image in the first window
    window_name1 = 'Original Image'

    cv2.namedWindow(window_name1, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name1, original)
    cv2.waitKey(5000)

    sobel(original)
    canny(original)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image improvement via area-to-pixel filters')
    parser.add_argument('-i', '--input', help='Path to input image.', default='cheerios.png')
    args = parser.parse_args()

    main(args)