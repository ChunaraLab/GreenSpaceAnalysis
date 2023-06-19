import itertools
import os
import cv2
import sys
import numpy as np
from tqdm import tqdm
from shapely.geometry import shape


def tint_image(image_garden_mask, image_garden_rgb):
    greyscale_image = cv2.cvtColor(image_garden_rgb, cv2.COLOR_BGR2GRAY).astype(np.float)
    greyscale_image = np.reshape(greyscale_image, (image_garden_mask.shape[0], image_garden_mask.shape[1], 1))

    tinted_image = np.concatenate((greyscale_image, greyscale_image, greyscale_image), axis=2)
    tinted_image = np.clip(tinted_image * 2 + 80, 0, 255)
    tinted_image = tinted_image.astype(np.uint8)

    return tinted_image


def debug_output_garden(tile_file_name, image_garden_bgr, image_garden_mask, downsampled_image_scale=1):

    downsampled_shape = (int(image_garden_bgr.shape[1] / downsampled_image_scale),
                         int(image_garden_bgr.shape[0] / downsampled_image_scale))
    downsampled_image = cv2.resize(image_garden_bgr, dsize=downsampled_shape, interpolation=cv2.INTER_AREA)
    cv2.imwrite(tile_file_name + '-1-garden.png', downsampled_image)

    tinted_image = tint_image(image_garden_mask, image_garden_bgr)[:, :, 0:3]
    invert_mask = np.invert(image_garden_mask)
    masked_image = image_garden_bgr.copy()[:, :, 0:3]
    masked_image[invert_mask] = tinted_image[invert_mask]
    downsampled_image = cv2.resize(masked_image, dsize=downsampled_shape, interpolation=cv2.INTER_AREA)
    cv2.imwrite(tile_file_name + '-2-maskedGarden.png', downsampled_image)

    mask = (image_garden_mask * 255).astype(np.uint8)
    downsampled_mask = cv2.resize(mask, dsize=downsampled_shape, interpolation=cv2.INTER_AREA)
    black = np.zeros([downsampled_shape[1], downsampled_shape[0]], dtype=np.uint8)
    white = np.full([downsampled_shape[1], downsampled_shape[0]], fill_value=255, dtype=np.uint8)
    downsampled_mask_binary = np.where(downsampled_mask > 127, white, black).reshape((downsampled_shape[1],
                                                                                      downsampled_shape[0], 1))
    downsampled_mask_bgr = np.concatenate((downsampled_mask_binary, downsampled_mask_binary, downsampled_mask_binary),
                                          axis=2)
    cv2.imwrite(tile_file_name + '-2-mask.png', downsampled_mask_bgr)


def apply_vegetation_index(bgr_index, bgr_image, mask, downsampled_image_scale=1, only_paint_vegetation=True):
    """
    applies an index function (which takes b,g,r tuples as input) and passes it over all the pixels in the image

    :param bgr_index: vegetation index object to be used
    :param bgr_image: image to be analysed, in the form [width, height, 3] with each value in the range 0..255, with
            the colours ordered blue, green, red
    :param mask: mask to determine which pixels to analyse; mask is a boolean array matching size with supplied img.
            Note that 'True' indicates a pixel is to be ignored.
    :param downsampled_image_scale: how to downsample images if output (e.g. 1=100% scale, 2=50% scale)
    :param only_paint_vegetation: should we paint the entire garden or just the estimated vegetation?
    :return: average vegetation index over the masked pixels, number of non-zero mask pixels, image with non-zero
            pixels painted
    """
    # Change here !!! 
    # num_pixels_in_polygon = np.count_nonzero(mask)
    num_pixels_in_polygon = bgr_image.shape[0]*bgr_image.shape[1]
    print("****** num_pixels_in_polygon ******")
    print(num_pixels_in_polygon)
    if num_pixels_in_polygon == 0:
        return 0.0, 0, bgr_image

    vegetation_presence = bgr_index.index(bgr_image)
    # print("****** vegetation_presence ******")
    # np.set_printoptions(threshold=sys.maxsize)
    # print(vegetation_presence)
    # print(np.count_nonzero(vegetation_presence))
    print("Outputing vegetation indices...")

    zeros = np.zeros(shape=bgr_image.shape[:2], dtype=np.float)

    # Change here!!! I commented them out
    # if downsampled_image_scale > 0:
    #     channel_shape = (bgr_image.shape[0], bgr_image.shape[1], 1)

    #     r = zeros.astype(np.uint8).reshape(channel_shape)
    #     g = np.where(mask, vegetation_presence * 255, zeros).astype(np.uint8).reshape(channel_shape)
    #     b = zeros.astype(np.uint8).reshape(channel_shape)

    #     image_index_bgr = np.concatenate((b, g, r), axis=2)

    #     tinted_image = tint_image(mask, bgr_image)

    #     mask_3d = mask.reshape((mask.shape[0], mask.shape[1], 1))
    #     bgr_mask = np.concatenate((mask_3d, mask_3d, mask_3d), axis=2)

    #     if only_paint_vegetation:
    #         vegetation_indices_x_by_y_by_1 = vegetation_presence.reshape(channel_shape)
    #         bgr_image_with_green_overlay = np.where(vegetation_indices_x_by_y_by_1, image_index_bgr,
    #                                                 bgr_image[:, :, 0:3])
    #         image_combined = np.where(bgr_mask, bgr_image_with_green_overlay, tinted_image)
    #     else:
    #         image_combined = np.where(bgr_mask, image_index_bgr, tinted_image)

    #     downsampled_shape = (int(image_combined.shape[1] / downsampled_image_scale),
    #                          int(image_combined.shape[0] / downsampled_image_scale))
    #     downsampled_image = cv2.resize(image_combined, dsize=downsampled_shape, interpolation=cv2.INTER_AREA)
    # else:
    #     downsampled_image = None
    downsampled_image = None

    # masked_vegetation_presence = np.where(mask, vegetation_presence, zeros)
    # total_num_vegetation_pixels = masked_vegetation_presence.sum()
    total_num_vegetation_pixels = vegetation_presence.sum()

    #    np.nan_to_num(masked_vegetation_indices, copy=False)
    fraction_of_vegetation_present = total_num_vegetation_pixels / num_pixels_in_polygon

    return fraction_of_vegetation_present, num_pixels_in_polygon, downsampled_image, vegetation_presence


def calculate_feature_indices(map_loader, json_features, vegetation_indices, output_folder_name,
                              downsampled_image_scale=0, only_paint_vegetation=False):
    if downsampled_image_scale > 0:
        os.makedirs(output_folder_name, exist_ok=True)

    feature_geometries = {}
    tile_geometries = {}
    progress_bar = tqdm(json_features, unit='feature', desc='Sorting features', ascii=True)

    for feature in progress_bar:
        if "id" in feature['properties']:
            feature_id = feature['properties']['id']
        else:
            feature_id = feature['properties']['toid']

        feature_uprn = feature['properties']['uprn']

        geometry = shape(feature['geometry'])

        garden_centre = (geometry.centroid.x, geometry.centroid.y)


        # print("********* Geometry Info *********")
        # print(geometry)
        # print(garden_centre)

        tile_geometry = map_loader.calculate_tile_geometry(geometry)
        # print("********* tile_geometry *********")
        # print(tile_geometry)
        tile_geometries[feature_id] = tile_geometry
        # print(tile_geometries)

        feature_tile_coord = (int(tile_geometry.bounds[0]) * 1000) + int(tile_geometry.bounds[1])

        existing_geometries = feature_geometries.get(feature_tile_coord, [])
        existing_geometries += [(feature_id, feature_uprn, garden_centre)]
        feature_geometries[feature_tile_coord] = existing_geometries

        # print("********* feature_geometries *********")
        # print(feature_geometries)

    sorted_tile_coord_keys = sorted(feature_geometries.keys())

    feature_info_lists_in_tile_order = [feature_geometries[k] for k in sorted_tile_coord_keys]
    feature_info_in_tile_order = list(itertools.chain.from_iterable(feature_info_lists_in_tile_order))

    # print("********* feature_info_in_tile_order *********")
    # print(feature_info_in_tile_order)

    feature_indices = []
    progress_bar = tqdm(feature_info_in_tile_order, unit='feature', desc='Analysing features', ascii=True)
    for feature_info in progress_bar:
        feature_id, feature_uprn, garden_centroid = feature_info

        # Change here!!! Tile geometry doesn't work. Using original geometry
        # geometry = tile_geometries[feature_id]

        # print("********* geometry *********")
        # print(geometry)

        if downsampled_image_scale > 0:
            debug_base_file_name = os.path.join(output_folder_name, feature_id + '-' + map_loader.name)
        else:
            debug_base_file_name = None

        imgs, mask, surface_area_m2, debug_tile_name, image_filenames = map_loader.get_image_and_mask(geometry, debug_base_file_name)

        vegetation_results = []

        step_count = 3
        for vegetation_index in vegetation_indices:
            try:
                # print("**** Before apply_vegetation_index ****")
                # print("**** Img is ****")
                # print(img.shape)
                # print("**** mask is ****")
                # print(mask.shape)
                # print("**** downsampled_image_scale is ****")
                # print(downsampled_image_scale)
                # print("**** only_paint_vegetation is ****")
                # print(only_paint_vegetation)
                
                vegetation_presences = []
                for img in imgs:
                    fraction_of_vegetation_present, num_pixels_in_polygon, vegetation_image, vegetation_presence \
                        = apply_vegetation_index(vegetation_index, img, mask, downsampled_image_scale,
                                               only_paint_vegetation)
                    vegetation_presences.append(vegetation_presence)

                    # if downsampled_image_scale > 0:
                    #     cv2.imwrite(f'{debug_tile_name}-{step_count}-{vegetation_index.short_name}.png', vegetation_image)

                    step_count += 1

                    vegetation_results.append((fraction_of_vegetation_present, num_pixels_in_polygon, vegetation_image))

            except Exception as e:
                print()
                print(f'Failed "{vegetation_index.short_name}" index with feature_id="{feature_id}"')
                print()
                raise e

        feature_indices.append((feature_id, feature_uprn, garden_centroid, surface_area_m2, vegetation_results))

    progress_bar.close()
    print("******* Returned results from calculate_indices *******")
    # print("[feature_id, feature_uprn, garden_centroid, surface_area_m2, [fraction_of_vegetation_present, num_pixels_in_polygon, vegetation_image]]")
    # print(feature_indices)
    return feature_indices, vegetation_presences, image_filenames
