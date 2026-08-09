import pandas as pd
import os
import cv2
import argparse
import random
from sklearn.model_selection import train_test_split

def load_and_clean(dataset_dir):

    exp1 = pd.read_csv(
        os.path.join(dataset_dir, "labels_exp1.csv")
    )

    exp2 = pd.read_csv(
        os.path.join(dataset_dir, "labels_exp2.csv")
    )

    exp1["uuid"] = exp1["uuid"].astype(str)
    exp2["uuid"] = exp2["uuid"].astype(str)

    # Patients appearing in both experiments
    overlap = set(exp1["uuid"]) & set(exp2["uuid"])

    print(
        f"Patients present in both EXP1 and EXP2: "
        f"{len(overlap)}"
    )

    if len(overlap) > 0:
        print(
            f"Keeping overlap in EXP1 and "
            f"removing from EXP2"
        )

 
    exp2 = exp2[
        ~exp2["uuid"].isin(overlap)
    ]

    print(
        f"EXP1 patients: "
        f"{exp1['uuid'].nunique()}"
    )

    print(
        f"EXP2 patients after cleanup: "
        f"{exp2['uuid'].nunique()}"
    )

    merged_df = pd.concat(
        [exp1, exp2],
        ignore_index=True
    )

    print(
        f"Total unique patients after merge: "
        f"{merged_df['uuid'].nunique()}"
    )

    return merged_df

def build_real_pool(processed_dir, allowed_uuids=None):
    """
    Build real slice pool only from specified patients.
    This prevents patient leakage.
    """

    real_pool = []

    allowed_uuids = (
        set(map(str, allowed_uuids))
        if allowed_uuids is not None
        else None
    )

    for uuid in os.listdir(processed_dir):

        scan_path = os.path.join(processed_dir, str(uuid))

        if not os.path.isdir(scan_path):
            continue

        # FIX: only keep patients belonging to this split
        if allowed_uuids is not None and str(uuid) not in allowed_uuids:
            continue

        for file in os.listdir(scan_path):
            if file.endswith(".jpg"):
                real_pool.append((str(uuid), file))

    return real_pool



def save_splits(
        df,
        processed_dir,
        output_dir,
        offset_start,
        offset_end,
        split_name,
        real_pool):

    saved = set()

    deepfake_count = 0
    real_used = 0


    for _, row in df.iterrows():

        uuid = str(row["uuid"])
        img_type = row["type"]
        slice_no = int(row["slice"])

        for offset in range(offset_start, offset_end + 1):

            cur_slice = slice_no + offset
            key = (uuid, cur_slice)

            if key in saved:
                continue

            saved.add(key)

            img_path = os.path.join(
                processed_dir,
                uuid,
                f"slice_{cur_slice}.jpg"
            )

            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (224, 224))

            label = (
                "Deepfake"
                if img_type in ["FM", "FB"]
                else "Real"
            )

            if label == "Deepfake":

                out_dir = os.path.join(
                    output_dir,
                    split_name,
                    "Deepfake"
                )

                os.makedirs(out_dir, exist_ok=True)

                filename = f"{uuid}_slice{cur_slice}.png"

                cv2.imwrite(
                    os.path.join(out_dir, filename),
                    img
                )

                deepfake_count += 1

    print(f"{split_name} Deepfake count: {deepfake_count}")

    random.shuffle(real_pool)

    real_target = deepfake_count

    if len(real_pool) < real_target:
        print(
            f"WARNING: only {len(real_pool)} real slices "
            f"available for {split_name}"
        )

    selected_real = real_pool[:real_target]

    for uuid, file in selected_real:

        img_path = os.path.join(
            processed_dir,
            uuid,
            file
        )

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (224, 224))

        out_dir = os.path.join(
            output_dir,
            split_name,
            "Real"
        )

        os.makedirs(out_dir, exist_ok=True)

        filename = f"REAL_{uuid}_{file}"

        cv2.imwrite(
            os.path.join(out_dir, filename),
            img
        )

        real_used += 1

    print(f"{split_name} Real count (balanced): {real_used}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--offset_start",
        type=int,
        default=-10
    )

    parser.add_argument(
        "--offset_end",
        type=int,
        default=10
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    random.seed(args.seed)


    df = load_and_clean(args.dataset_dir)

    uuids = df["uuid"].astype(str).unique()

    train_ids, test_ids = train_test_split(
        uuids,
        test_size=args.test_size,
        random_state=args.seed
    )

    train_ids = set(map(str, train_ids))
    test_ids = set(map(str, test_ids))

    # sanity check
    overlap = train_ids.intersection(test_ids)

    print(f"Train patients: {len(train_ids)}")
    print(f"Test patients: {len(test_ids)}")
    print(f"Patient overlap: {len(overlap)}")

    assert len(overlap) == 0, \
        "Leakage detected: train/test patient overlap"

    train_df = df[
        df["uuid"].astype(str).isin(train_ids)
    ]

    test_df = df[
        df["uuid"].astype(str).isin(test_ids)
    ]


    train_real_pool = build_real_pool(
        args.processed_dir,
        train_ids
    )

    test_real_pool = build_real_pool(
        args.processed_dir,
        test_ids
    )

    print(
        f"Train real candidates: "
        f"{len(train_real_pool)}"
    )

    print(
        f"Test real candidates: "
        f"{len(test_real_pool)}"
    )


    save_splits(
        train_df,
        args.processed_dir,
        args.output_dir,
        args.offset_start,
        args.offset_end,
        "Train",
        train_real_pool
    )

    save_splits(
        test_df,
        args.processed_dir,
        args.output_dir,
        args.offset_start,
        args.offset_end,
        "Test",
        test_real_pool
    )


if __name__ == "__main__":
    main()
