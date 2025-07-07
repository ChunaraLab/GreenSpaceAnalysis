def custom_transforms(image, mask):
    b = image
    hsvb = rgb2hsv(b)
    hsvb_old = hsvb[:, :, 0]
    hsvb_new = hsvb[:, :, 0]
            
    val_max, val_min = 0.42, 0.05
    # variation = (val_max - val_min)/2
    # std_dev = variation/3
    std_dev = 0.5
    jitter = np.random.normal(0, std_dev)
    up, down = 0.42, 0.05

    if (jitter > 0):
        up, down = up-jitter, down
        hsvb_new = np.where(np.logical_and(hsvb_new<=val_max,hsvb_new>up), val_max, hsvb_new)
        hsvb_new = np.where(np.logical_and(hsvb_new<=up,hsvb_new>=down), hsvb_new+jitter, hsvb_new)
            
    if (jitter < 0):
        up, down = up, down-jitter
        hsvb_new = np.where(np.logical_and(hsvb_new>=val_min,hsvb_new<down), val_min, hsvb_new)
        hsvb_new = np.where(np.logical_and(hsvb_new>=down, hsvb_new<=up), hsvb_new+jitter, hsvb_new)
            
    mask_green = np.where(mask > 0, 1, 0)
    mask_no_green = np.where(mask == 0, 1, 0)
            
    hsvb_new = np.add(np.multiply(hsvb_new, mask_green), np.multiply(hsvb_old, mask_no_green))
            
    hsvb[:, :, 0] = hsvb_new
            
    norm_image = cv2.normalize(hsv2rgb(hsvb),None,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
            return norm_image
            
