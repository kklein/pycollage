import logging
import pickle
from pathlib import Path

import click
from annoy import AnnoyIndex
from PIL import Image, ImageStat
from tqdm import tqdm, trange

_COMPONENT_SIZE = 20
_DICT_FILENAME = "id_to_file.pkl"
_N_DIMS = 3
_INDEX_FILENAME = "index.ann"
_DISTANCE = "euclidean"


@click.group()
def cli():
    pass


def resize_collection(path: Path, component_size: int = _COMPONENT_SIZE) -> Path:
    new_path = path.parent / "resized_images"
    new_path.mkdir(exist_ok=False)
    files = list(path.iterdir())
    for file in tqdm(files, desc="resizing"):
        if not file.is_file():
            continue
        image = Image.open(file.absolute())

        resized_image = image.resize((component_size, component_size), Image.ANTIALIAS)
        resized_image.save(new_path / f"{file.stem}_resized.png", "PNG", quality=90)
    return new_path


def build_index(
    path: Path,
    n_trees: int = 10,
    n_dims: int = _N_DIMS,
    distance: str = _DISTANCE,
    save: bool = True,
) -> tuple[AnnoyIndex, dict[int, Path]]:
    index = AnnoyIndex(n_dims, distance)
    id_to_file = {}

    image_id = 0
    files = list(path.iterdir())
    for file in tqdm(files, desc="building index"):
        if not file.is_file():
            continue
        id_to_file[image_id] = file
        image = Image.open(file.absolute())
        stat = ImageStat.Stat(image)
        index.add_item(image_id, stat.mean[:3])
        image_id += 1

    index.build(n_trees)

    if save:
        index_path = path.parent / "index"
        index_path.mkdir(exist_ok=False)
        with open(index_path / _DICT_FILENAME, "wb") as filehandle:
            pickle.dump(id_to_file, filehandle)
        index.save(str(index_path / _INDEX_FILENAME))
        print(f"Saved index data to {index_path}.")
    
    return index, id_to_file


def _retrieve(
    r: int, g: int, b: int, index: AnnoyIndex, id_to_file: dict[int, Path]
) -> Image:
    retrieved_id = index.get_nns_by_vector([r, g, b], 1)[0]
    return Image.open(id_to_file[retrieved_id])


def _get_component_size(id_to_file: dict[int, Path]) -> int:
    path = next(iter(id_to_file.values()))
    random_image = Image.open(path)
    width = random_image._size[0]
    height = random_image._size[1]
    if width != height:
        raise ValueError(
            "Component images are not of a square size. Their width is "
            f"{width} while their height is {height}."
            "Have you provided the correct path?"
        )
    return width


def build_collage(
    target_image_path: Path, index: AnnoyIndex, id_to_file: dict[int, Path]
) -> Image:
    target_image = Image.open(target_image_path)
    component_size = _get_component_size(id_to_file)
    collage = Image.new(
        "RGBA",
        (
            target_image._size[0] * component_size,
            target_image._size[1] * component_size,
        ),
        color=(255, 255, 255, 255),
    )

    for x in trange(target_image._size[0], desc="columns", position=0):
        for y in trange(target_image._size[1], desc="rows", position=1, leave=False):
            r, g, b = target_image.getpixel((x, y))[:3]
            retrieved_image = _retrieve(
                r,
                g,
                b,
                index,
                id_to_file,
            )
            collage.paste(retrieved_image, (x * component_size, y * component_size))

    collage.show()
    target_image.show()
    return collage


# TODO: Add overwrite option
@cli.command("prepare-collection")
@click.argument(
    "collection_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option("--component_size", type=int, default=_COMPONENT_SIZE)
def prepare_collection(collection_path, component_size=_COMPONENT_SIZE):
    path = Path(collection_path)
    resized_path = resize_collection(path, component_size)
    build_index(resized_path)


@cli.command("build")
@click.argument(
    "index_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument(
    "target_image", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
def build(index_path, target_image):
    path = Path(index_path)
    with open(path / _DICT_FILENAME, "rb") as filehandle:
        id_to_file = pickle.load(filehandle)
    index = AnnoyIndex(_N_DIMS, _DISTANCE)
    index.load(str(path / _INDEX_FILENAME))
    build_collage(Path(target_image), index, id_to_file)
