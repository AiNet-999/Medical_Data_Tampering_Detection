import numpy as np
import os
import cv2
import pydicom
import pandas as pd
import argparse

def load_tampered_data(csv_path):
    tampered_df = pd.read_csv(csv_path)
    tampered_df['uuid'] = tampered_df['uuid'].astype(str)
    tampered_df['slice'] = tampered_df['slice'].astype(int)
    return tampered_df

def is_tampered(tampered_df, scan_uuid, slice_idx):
    scan_uuid_str = str(scan_uuid)
    tampered_slices = tampered_df[tampered_df['uuid'] == scan_uuid_str]
    tampered_slices_in_scan = tampered_slices[tampered_slices['slice'] == slice_idx]
    if not tampered_slices_in_scan.empty:
        return True, tampered_slices_in_scan[['x', 'y']].values.tolist()
    return False, []

def save_slices(scan_uuid, scan_data, dicom_slices, tampered_df, output_base_dir, annotate):
    output_dir = os.path.join(output_base_dir, scan_uuid)
    os.makedirs(output_dir, exist_ok=True)

    for slice_idx in range(scan_data.shape[0]):
        dicom = dicom_slices[slice_idx]
        raw_image = scan_data[slice_idx, :, :]
        slice_img = cv2.normalize(raw_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        tampered, tampered_positions = is_tampered(tampered_df, scan_uuid, slice_idx)

        if tampered and annotate:
            for x, y in tampered_positions:
                if 0 <= x < slice_img.shape[1] and 0 <= y < slice_img.shape[0]:
                    cv2.circle(slice_img, (x, y), radius=23, color=(255, 255, 255), thickness=3)
            slice_filename = os.path.join(output_dir, f'tampered_slice_{slice_idx}.jpg')
        else:
            slice_filename = os.path.join(output_dir, f'slice_{slice_idx}.jpg')

        cv2.imwrite(slice_filename, slice_img)

    print(f"All slices saved in folder: {output_dir}")

def load_dicom(path2scan_dir):
    dcms = os.listdir(path2scan_dir)
    first_slice_data = pydicom.dcmread(os.path.join(path2scan_dir, dcms[0]))
    first_slice = first_slice_data.pixel_array
    spacing_xy = np.array(first_slice_data.PixelSpacing, dtype=float)
    spacing_z = np.float64(first_slice_data.SliceThickness)
    spacing = np.array([spacing_z, spacing_xy[1], spacing_xy[0]])
    scan = np.zeros((len(dcms), first_slice.shape[0], first_slice.shape[1]))
    raw_slices = []
    indexes = []

    for dcm in dcms:
        slice_data = pydicom.dcmread(os.path.join(path2scan_dir, dcm))
        slice_data.filename = dcm
        raw_slices.append(slice_data)
        indexes.append(float(slice_data.ImagePositionPatient[2]))

    indexes = np.array(indexes, dtype=float)
    raw_slices = [x for _, x in sorted(zip(indexes, raw_slices))]

    for i, slice in enumerate(raw_slices):
        scan[i, :, :] = slice.pixel_array

    return scan, spacing, raw_slices

def process_dataset(csv_path, scan_folder, output_dir, annotate):
    tampered_df = load_tampered_data(csv_path)
    files = os.listdir(scan_folder)
    for f in files:
        scan_uuid = f.split('.')[0]
        scan_path = os.path.join(scan_folder, f)
        scan, spacing, raw_slices = load_dicom(scan_path)
        print(f'Processing {scan_uuid} with shape {scan.shape}')
        save_slices(scan_uuid, scan, raw_slices, tampered_df, output_dir, annotate)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True, help='Root path containing labels_exp*.csv and CT_Scans/* folders')
    parser.add_argument('--output_dir', required=True, help='Output directory to save results')
    parser.add_argument('--annotate', action='store_true', help='Annotate tampered regions with circles if set')
    args = parser.parse_args()

    datasets = [
        {
            'csv_path': os.path.join(args.dataset_dir, 'labels_exp1.csv'),
            'scan_folder': os.path.join(args.dataset_dir, 'CT_Scans', 'EXP1_blind')
        },
        {
            'csv_path': os.path.join(args.dataset_dir, 'labels_exp2.csv'),
            'scan_folder': os.path.join(args.dataset_dir, 'CT_Scans', 'EXP2_open')
        }
    ]

    for dataset in datasets:
        process_dataset(
            csv_path=dataset['csv_path'],
            scan_folder=dataset['scan_folder'],
            output_dir=args.output_dir,
            annotate=args.annotate
        )

if __name__ == '__main__':
    main()
