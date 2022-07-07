import cv2
import pathlib

thisdir = pathlib.Path(__file__).resolve().parent

def main():
    savepath = thisdir.joinpath("outputs", "networks.avi")

    image_paths = sorted(
        thisdir.joinpath("outputs", "networks").glob("*.png"),
        key=lambda x: int(pathlib.Path(x).stem.split("_")[-1])
    )

    frame = cv2.imread(str(image_paths[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(str(savepath), 0, 2, (width, height))

    for image_path in image_paths:
        video.write(cv2.imread(str(image_path)))

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    main()