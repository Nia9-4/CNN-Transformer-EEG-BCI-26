from mne.datasets import eegbci

def download():
    BAD = {88, 89, 92, 100}
    runs = [4, 8, 12]
    print("Starting mass download... this may take a while.")

    for s in range(1, 110):
        if s in BAD:
            continue
        try:
            eegbci.load_data(subjects=s, runs=runs, update_path=True)
            print(f"Downloaded subject {s}")
        except Exception as e:
            print(f"Failed to download subject {s}: {e}")

    print("All available data downloaded to ~/mne_data")

if __name__ == "__main__":
    download()