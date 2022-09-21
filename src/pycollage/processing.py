import logging
from pathlib import Path

import click
from annoy import AnnoyIndex
from pillow import Image, ImageStat
from tqdm import tqdm, trange

COMPONENT_SIZE = 20


@click.group()
def cli():
    pass


@cli.command("resize-collection")
@click.argument(
    "collection_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
def resize_collection(
    collection_path: str, component_size: int = COMPONENT_SIZE
) -> None:
    path = Path(collection_path)
    new_path = path.parent / "resized_images"
    new_path.mkdir(exist_ok=False)
    files = list(path.iterdir())
    for file in tqdm(files):
        if not file.is_file():
            continue
        image = Image.open(file.absolute())

        resized_image = image.resize((component_size, component_size), Image.ANTIALIAS)
        resized_image.save(new_path / f"{file.stem}_resized.png", "PNG", quality=90)


def build_index(
    path: Path, n_trees: int = 10, output_file: str = "index.ann"
) -> tuple[AnnoyIndex, dict[int, Path]]:
    n_dims = 3
    index = AnnoyIndex(n_dims, "euclidean")
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
    index.save(output_file)
    logging.info(f"Saved index to {output_file}.")
    return index, id_to_file


def _retrieve(
    r: int, g: int, b: int, index: AnnoyIndex, id_to_file: dict[int, Path]
) -> Image:
    retrieved_id = index.get_nns_by_vector([r, g, b], 1)[0]
    return Image.open(id_to_file[retrieved_id])


def build_collage(
    target_image_path: Path, index: AnnoyIndex, id_to_file: dict[int, Path]
) -> Image:
    target_image = Image.open(target_image_path)
    collage = Image.new(
        "RGBA",
        (
            target_image._size[0] * COMPONENT_SIZE,
            target_image._size[1] * COMPONENT_SIZE,
        ),
        color=(255, 255, 255, 255),
    )

    for x in trange(target_image._size[0], desc="outer", position=0):
        for y in trange(target_image._size[1], desc="inner", position=1, leave=False):
            r, g, b = target_image.getpixel((x, y))[:3]
            retrieved_image = _retrieve(
                r,
                g,
                b,
                index,
                id_to_file,
            )
            collage.paste(retrieved_image, (x * COMPONENT_SIZE, y * COMPONENT_SIZE))

    collage.show()
    target_image.show()
    return collage


@cli.command("build")
@click.argument(
    "component_collection", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument(
    "target_image", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
def build(component_collection, target_image):
    index, id_to_file = build_index(Path(component_collection))
    build_collage(Path(target_image), index, id_to_file)


if __name__ == "__main__":
    index, id_to_file = build_index(
        Path("/users/kevin/Code/collage/data/resized_images/")
    )
    build_collage(Path("/users/kevin/Downloads/tomato.jpeg"), index, id_to_file)
