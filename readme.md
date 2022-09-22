Create collages of images.

![](tomato.png)

## Installation

Install the package via either of these options:
1. ```console
   $ git clone https://github.com/kklein/pycollage.git
   $ conda env create -f environment.yml
   $ conda activate pycollage
   $ pip install --no-build-isolation -e .
   ```
2. `$ pip install pycollage`
3. `$ conda install pycollage -c conda-forge`

## Usage

In order to use `pycollage`, you should have a collection of images
which makes up for the individual components of the collage as well
as a target image which the collage seeks to imitate.

In the following we will assume that you saved the image collection under
`/users/Anne/image_collection` and the target image under
`/users/Anne/target_image.png`. Best use absolute paths.

### Processing the collection

The first step consists of processing the image collection. In order to do so,
run

```
$ pycollage process-collection /users/Anne/image_collection
```

By default, this will resize the images to each be 20x20 pixels. If you'd like
a different component size, e.g. 30x30 pixels, run

```
$ pycollage process-collection /users/Anne/image_collection --component_size 30
```

instead.

You will notice that executing this will tell you where an index directory has been
created. Given the paths from above, this should be `/users/Anne/index`. Moreover
another new directory has been created: `/users/Anne/image_collection_resized`.

### Building a collage

Now that the collection has been duely processed, we can embark onto actually building a collage.

For that purpose, simply run

```
$ pycollage build /users/Anne/index /users/Anne/target_image
```

where the first path corresponds to the path that was mentioned to you
when executing `process-collection`.

At the end of the process both the original target image as well as the collage
imitating the target image will be opened. You can then save the collage to wherever
you like.

### Runtime
Note that this process will take longer
* the larger the `component_size` you provide
* the larger your collection of images
* the larger the resolution of the target image

Since the collage will scale up the original target image by `component_size` per
dimensions - i.e., 20 by default - you might want to scale down your original
target image to below 1000x1000 pixels.

### Troubleshooting
`pycollage` relies on [annoy](https://github.com/spotify/annoy) under the hood. Hence when trying to install
`pycollage`, annoy will be attempted to be installed as well. Please check
their [issues page](https://github.com/spotify/annoy/issues) to figure out more
about possible installation problems and potential fixes.
