"""
Pipeline modules for frame selection.
"""

import sys
import time
import math
import warnings
import multiprocessing as mp

from typing import Union, Tuple

import numpy as np
from skimage.measure import compare_ssim, compare_nrmse

from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.image import crop_image, pixel_distance, center_pixel, create_mask
from pynpoint.util.module import progress, memory_frames, locate_star
from pynpoint.util.remove import write_selected_data, write_selected_attributes


class RemoveFramesModule(ProcessingModule):
    """
    Pipeline module for removing images by their index number.
    """

    def __init__(self,
                 frames,
                 name_in='remove_frames',
                 image_in_tag='im_arr',
                 selected_out_tag='im_arr_selected',
                 removed_out_tag='im_arr_removed'):
        """
        Parameters
        ----------
        frames : str, list, tuple, range, or numpy.ndarray
            A tuple or array with the frame indices that have to be removed or a database tag
            pointing to a list of frame indices.
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        selected_out_tag : str
            Tag of the database entry with the remaining images after removing the specified
            images. Should be different from *image_in_tag*. No data is written when set to
            *None*.
        removed_out_tag : str
            Tag of the database entry with the images that are removed. Should be different
            from *image_in_tag*. No data is written when set to *None*.

        Returns
        -------
        NoneType
            None
        """

        super(RemoveFramesModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        if selected_out_tag is None:
            self.m_selected_out_port = None
        else:
            self.m_selected_out_port = self.add_output_port(selected_out_tag)

        if removed_out_tag is None:
            self.m_removed_out_port = None
        else:
            self.m_removed_out_port = self.add_output_port(removed_out_tag)

        if isinstance(frames, str):
            self.m_index_in_port = self.add_input_port(frames)
        else:
            self.m_index_in_port = None

            if isinstance(frames, (tuple, list, range)):
                self.m_frames = np.asarray(frames, dtype=np.int)

            elif isinstance(frames, np.ndarray):
                self.m_frames = frames

    def _initialize(self):

        if self.m_selected_out_port is not None:
            if self.m_image_in_port.tag == self.m_selected_out_port.tag:
                raise ValueError('Input and output ports should have a different tag.')

        if self.m_removed_out_port is not None:
            if self.m_image_in_port.tag == self.m_removed_out_port.tag:
                raise ValueError('Input and output ports should have a different tag.')

        if self.m_index_in_port is not None:
            self.m_frames = self.m_index_in_port.get_all()

        if np.size(np.where(self.m_frames >= self.m_image_in_port.get_shape()[0])) > 0:
            raise ValueError(f'Some values in \'frames\' are larger than the total number of '
                             f'available frames, {self.m_image_in_port.get_shape()[0]}')

        if self.m_selected_out_port is not None:
            self.m_selected_out_port.del_all_data()
            self.m_selected_out_port.del_all_attributes()

        if self.m_removed_out_port is not None:
            self.m_removed_out_port.del_all_data()
            self.m_removed_out_port.del_all_attributes()

    def run(self):
        """
        Run method of the module. Removes the frames and corresponding attributes, updates the
        NFRAMES attribute, and saves the data and attributes.

        Returns
        -------
        NoneType
            None
        """

        self._initialize()

        memory = self._m_config_port.get_attribute('MEMORY')

        nimages = self.m_image_in_port.get_shape()[0]
        frames = memory_frames(memory, nimages)

        if memory == 0 or memory >= nimages:
            memory = nimages

        start_time = time.time()
        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), 'Running RemoveFramesModule...', start_time)

            images = self.m_image_in_port[frames[i]:frames[i+1], ]

            index_del = np.where(np.logical_and(self.m_frames >= frames[i], \
                                                self.m_frames < frames[i+1]))

            write_selected_data(images,
                                self.m_frames[index_del]%memory,
                                self.m_selected_out_port,
                                self.m_removed_out_port)

        sys.stdout.write('Running RemoveFramesModule... [DONE]\n')
        sys.stdout.flush()

        history = 'frames removed = '+str(np.size(self.m_frames))

        if self.m_selected_out_port is not None:
            # Copy attributes before write_selected_attributes is used
            self.m_selected_out_port.copy_attributes(self.m_image_in_port)
            self.m_selected_out_port.add_history('RemoveFramesModule', history)

        if self.m_removed_out_port is not None:
            # Copy attributes before write_selected_attributes is used
            self.m_removed_out_port.copy_attributes(self.m_image_in_port)
            self.m_removed_out_port.add_history('RemoveFramesModule', history)

        write_selected_attributes(self.m_frames,
                                  self.m_image_in_port,
                                  self.m_selected_out_port,
                                  self.m_removed_out_port)

        self.m_image_in_port.close_port()


class FrameSelectionModule(ProcessingModule):
    """
    Pipeline module for frame selection.
    """

    def __init__(self,
                 name_in='frame_selection',
                 image_in_tag='im_arr',
                 selected_out_tag='im_arr_selected',
                 removed_out_tag='im_arr_removed',
                 index_out_tag=None,
                 method='median',
                 threshold=4.,
                 fwhm=0.1,
                 aperture=('circular', 0.2),
                 position=(None, None, 0.5)):
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        selected_out_tag : str
            Tag of the database entry with the selected images that are written as output. Should
            be different from *image_in_tag*. No data is written when set to None.
        removed_out_tag : str
            Tag of the database entry with the removed images that are written as output. Should
            be different from *image_in_tag*. No data is written when set to None.
        index_out_tag : str
            Tag of the database entry with the list of frames indices that are removed with the
            frames selection. No data is written when set to *None*.
        method : str
            Perform the sigma clipping with respect to the median or maximum aperture flux by
            setting the *method* to 'median' or 'max'.
        threshold : float
            Threshold in units of sigma for the frame selection. All images that are a *threshold*
            number of sigmas away from the median photometry will be removed.
        fwhm : float
            The full width at half maximum (FWHM) of the Gaussian kernel (arcsec) that is used to
            smooth the images before the brightest pixel is located. Should be similar in size to
            the FWHM of the stellar PSF. A fixed position, specified by *position*, is used when
            *fwhm* is set to None
        aperture : tuple(str, float, float)
            Tuple with the aperture properties for measuring the photometry around the location of
            the brightest pixel. The first element contains the aperture type ('circular',
            'annulus', or 'ratio'). For a circular aperture, the second element contains the
            aperture radius (arcsec). For the other two types, the second and third element are the
            inner and outer radii (arcsec) of the aperture. The position of the aperture has to be
            specified with *position* when *fwhm* is set to None.
        position : tuple(int, int, float)
            Subframe that is selected to search for the star. The tuple contains the center (pix)
            and size (arcsec) (pos_x, pos_y, size). Setting *position* to None will use the full
            image to search for the star. If *position=(None, None, size)* then the center of the
            image will be used.

        Returns
        -------
        NoneType
            None
        """

        super(FrameSelectionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)

        if index_out_tag is None:
            self.m_index_out_port = None
        else:
            self.m_index_out_port = self.add_output_port(index_out_tag)

        if selected_out_tag is None:
            self.m_selected_out_port = None
        else:
            self.m_selected_out_port = self.add_output_port(selected_out_tag)

        if removed_out_tag is None:
            self.m_removed_out_port = None
        else:
            self.m_removed_out_port = self.add_output_port(removed_out_tag)

        self.m_method = method
        self.m_fwhm = fwhm
        self.m_aperture = aperture
        self.m_threshold = threshold
        self.m_position = position

    def _initialize(self):
        if self.m_image_in_port.tag == self.m_selected_out_port.tag or \
                self.m_image_in_port.tag == self.m_removed_out_port.tag:
            raise ValueError('Input and output ports should have a different tag.')

        if self.m_index_out_port is not None:
            self.m_index_out_port.del_all_data()
            self.m_index_out_port.del_all_attributes()

        if self.m_selected_out_port is not None:
            self.m_selected_out_port.del_all_data()
            self.m_selected_out_port.del_all_attributes()

        if self.m_removed_out_port is not None:
            self.m_removed_out_port.del_all_data()
            self.m_removed_out_port.del_all_attributes()

    def run(self):
        """
        Run method of the module. Smooths the images with a Gaussian kernel, locates the brightest
        pixel in each image, measures the integrated flux around the brightest pixel, calculates
        the median and standard deviation of the photometry, and applies sigma clipping to remove
        low quality images.

        Returns
        -------
        NoneType
            None
        """

        def _get_aperture(aperture):
            if aperture[0] == 'circular':
                aperture = (0., aperture[1]/pixscale)

            elif aperture[0] == 'annulus' or aperture[0] == 'ratio':
                aperture = (aperture[1]/pixscale, aperture[2]/pixscale)

            return aperture

        def _get_starpos(fwhm, position):
            starpos = np.zeros((nimages, 2), dtype=np.int64)

            if fwhm is None:
                starpos[:, 0] = position[0]
                starpos[:, 1] = position[1]

            else:
                if position is None:
                    center = None
                    width = None

                else:
                    if position[0] is None and position[1] is None:
                        center = None
                    else:
                        center = position[0:2]

                    width = int(math.ceil(position[2]/pixscale))

                for i, _ in enumerate(starpos):
                    starpos[i, :] = locate_star(image=self.m_image_in_port[i, ],
                                                center=center,
                                                width=width,
                                                fwhm=int(math.ceil(fwhm/pixscale)))

            return starpos

        def _photometry(images, starpos, aperture):
            check_pos_in = any(np.floor(starpos[:]-aperture[1]) < 0.)
            check_pos_out = any(np.ceil(starpos[:]+aperture[1]) > images.shape[0])

            if check_pos_in or check_pos_out:
                phot = np.nan

            else:
                im_crop = crop_image(images, starpos, 2*int(math.ceil(aperture[1])))

                npix = im_crop.shape[0]

                x_grid = y_grid = np.linspace(-(npix-1)/2, (npix-1)/2, npix)
                xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
                rr_grid = np.sqrt(xx_grid*xx_grid+yy_grid*yy_grid)

                if self.m_aperture[0] == 'circular':
                    phot = np.sum(im_crop[rr_grid < aperture[1]])

                elif self.m_aperture[0] == 'annulus':
                    phot = np.sum(im_crop[(rr_grid > aperture[0]) & (rr_grid < aperture[1])])

                elif self.m_aperture[0] == 'ratio':
                    phot = np.sum(im_crop[rr_grid < aperture[0]]) / \
                        np.sum(im_crop[(rr_grid > aperture[0]) & (rr_grid < aperture[1])])

            return phot

        self._initialize()

        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        nimages = self.m_image_in_port.get_shape()[0]

        aperture = _get_aperture(self.m_aperture)
        starpos = _get_starpos(self.m_fwhm, self.m_position)

        phot = np.zeros(nimages)

        start_time = time.time()
        for i in range(nimages):
            progress(i, nimages, 'Running FrameSelectionModule...', start_time)

            images = self.m_image_in_port[i]
            phot[i] = _photometry(images, starpos[i, :], aperture)

        if self.m_method == 'median':
            phot_ref = np.nanmedian(phot)
        elif self.m_method == 'max':
            phot_ref = np.nanmax(phot)

        phot_std = np.nanstd(phot)

        index_rm = np.logical_or((phot > phot_ref+self.m_threshold*phot_std),
                                 (phot < phot_ref-self.m_threshold*phot_std))

        index_rm[np.isnan(phot)] = True

        indices = np.where(index_rm)[0]
        indices = np.asarray(indices, dtype=np.int)

        if np.size(indices) > 0:
            memory = self._m_config_port.get_attribute('MEMORY')
            frames = memory_frames(memory, nimages)

            if memory == 0 or memory >= nimages:
                memory = nimages

            for i, _ in enumerate(frames[:-1]):
                images = self.m_image_in_port[frames[i]:frames[i+1], ]

                index_del = np.where(np.logical_and(indices >= frames[i], \
                                                    indices < frames[i+1]))

                write_selected_data(images,
                                    indices[index_del]%memory,
                                    self.m_selected_out_port,
                                    self.m_removed_out_port)

        else:
            warnings.warn('No frames were removed.')

        history = 'frames removed = '+str(np.size(indices))

        if self.m_index_out_port is not None:
            self.m_index_out_port.set_all(np.transpose(indices))
            self.m_index_out_port.copy_attributes(self.m_image_in_port)
            self.m_index_out_port.add_attribute('STAR_POSITION', starpos, static=False)
            self.m_index_out_port.add_history('FrameSelectionModule', history)

        if self.m_selected_out_port is not None:
            # Copy attributes before write_selected_attributes is used
            self.m_selected_out_port.copy_attributes(self.m_image_in_port)

        if self.m_removed_out_port is not None:
            # Copy attributes before write_selected_attributes is used
            self.m_removed_out_port.copy_attributes(self.m_image_in_port)

        write_selected_attributes(indices,
                                  self.m_image_in_port,
                                  self.m_selected_out_port,
                                  self.m_removed_out_port)

        if self.m_selected_out_port is not None:
            indices_select = np.ones(nimages, dtype=bool)
            indices_select[indices] = False
            indices_select = np.where(indices_select)

            self.m_selected_out_port.add_attribute('STAR_POSITION',
                                                   starpos[indices_select],
                                                   static=False)

            self.m_selected_out_port.add_history('FrameSelectionModule', history)

        if self.m_removed_out_port is not None:
            self.m_removed_out_port.add_attribute('STAR_POSITION',
                                                  starpos[indices],
                                                  static=False)

            self.m_removed_out_port.add_history('FrameSelectionModule', history)

        sys.stdout.write('Running FrameSelectionModule... [DONE]\n')
        sys.stdout.flush()

        self.m_image_in_port.close_port()


class RemoveLastFrameModule(ProcessingModule):
    """
    Pipeline module for removing every NDIT+1 frame from NACO data obtained in cube mode. This
    frame contains the average pixel values of the cube.
    """

    def __init__(self,
                 name_in='remove_last_frame',
                 image_in_tag='im_arr',
                 image_out_tag='im_arr_last'):
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Should be different from
            *image_in_tag*.

        Returns
        -------
        NoneType
            None
        """

        super(RemoveLastFrameModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):
        """
        Run method of the module. Removes every NDIT+1 frame and saves the data and attributes.

        Returns
        -------
        NoneType
            None
        """

        if self.m_image_out_port.tag == self.m_image_in_port.tag:
            raise ValueError('Input and output port should have a different tag.')

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        ndit = self.m_image_in_port.get_attribute('NDIT')
        nframes = self.m_image_in_port.get_attribute('NFRAMES')
        index = self.m_image_in_port.get_attribute('INDEX')

        nframes_new = []
        index_new = []

        start_time = time.time()
        for i, item in enumerate(ndit):
            progress(i, len(ndit), 'Running RemoveLastFrameModule...', start_time)

            if nframes[i] != item+1:
                warnings.warn(f'Number of frames ({nframes[i]}) is not equal to NDIT+1.')

            frame_start = np.sum(nframes[0:i])
            frame_end = np.sum(nframes[0:i+1]) - 1

            nframes_new.append(nframes[i]-1)
            index_new.extend(index[frame_start:frame_end])

            images = self.m_image_in_port[frame_start:frame_end, ]
            self.m_image_out_port.append(images)

        nframes_new = np.asarray(nframes_new, dtype=np.int)
        index_new = np.asarray(index_new, dtype=np.int)

        sys.stdout.write('Running RemoveLastFrameModule... [DONE]\n')
        sys.stdout.flush()

        self.m_image_out_port.copy_attributes(self.m_image_in_port)

        self.m_image_out_port.add_attribute('NFRAMES', nframes_new, static=False)
        self.m_image_out_port.add_attribute('INDEX', index_new, static=False)

        history = 'frames removed = NDIT+1'
        self.m_image_out_port.add_history('RemoveLastFrameModule', history)

        self.m_image_out_port.close_port()


class RemoveStartFramesModule(ProcessingModule):
    """
    Pipeline module for removing a fixed number of images at the beginning of each cube. This can
    be useful for NACO data in which the background is significantly higher in the first several
    frames of a data cube.
    """

    def __init__(self,
                 frames=1,
                 name_in='remove_last_frame',
                 image_in_tag='im_arr',
                 image_out_tag='im_arr_first'):
        """
        Parameters
        ----------
        frames : int
            Number of frames that are removed at the beginning of each cube.
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Should be different from
            *image_in_tag*.

        Returns
        -------
        NoneType
            None
        """

        super(RemoveStartFramesModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_frames = int(frames)

    def run(self):
        """
        Run method of the module. Removes a constant number of images at the beginning of each cube
        and saves the data and attributes.

        Returns
        -------
        NoneType
            None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        if self.m_image_out_port.tag == self.m_image_in_port.tag:
            raise ValueError('Input and output port should have a different tag.')

        nframes = self.m_image_in_port.get_attribute('NFRAMES')
        index = self.m_image_in_port.get_attribute('INDEX')

        index_new = []

        if 'PARANG' in self.m_image_in_port.get_all_non_static_attributes():
            parang = self.m_image_in_port.get_attribute('PARANG')
            parang_new = []

        else:
            parang = None

        if 'STAR_POSITION' in self.m_image_in_port.get_all_non_static_attributes():
            star = self.m_image_in_port.get_attribute('STAR_POSITION')
            star_new = []

        else:
            star = None

        start_time = time.time()
        for i, _ in enumerate(nframes):
            progress(i, len(nframes), 'Running RemoveStartFramesModule...', start_time)

            frame_start = np.sum(nframes[0:i]) + self.m_frames
            frame_end = np.sum(nframes[0:i+1])

            if frame_start >= frame_end:
                raise ValueError('The number of frames in the original data cube is equal or '
                                 'smaller than the number of frames that have to be removed.')

            index_new.extend(index[frame_start:frame_end])

            if parang is not None:
                parang_new.extend(parang[frame_start:frame_end])

            if star is not None:
                star_new.extend(star[frame_start:frame_end])

            images = self.m_image_in_port[frame_start:frame_end, ]
            self.m_image_out_port.append(images)

        sys.stdout.write('Running RemoveStartFramesModule... [DONE]\n')
        sys.stdout.flush()

        self.m_image_out_port.copy_attributes(self.m_image_in_port)

        self.m_image_out_port.add_attribute('NFRAMES', nframes-self.m_frames, static=False)
        self.m_image_out_port.add_attribute('INDEX', index_new, static=False)

        if parang is not None:
            self.m_image_out_port.add_attribute('PARANG', parang_new, static=False)

        if star is not None:
            self.m_image_out_port.add_attribute('STAR_POSITION', np.asarray(star_new), static=False)

        history = 'frames removed = '+str(self.m_frames)
        self.m_image_out_port.add_history('RemoveStartFramesModule', history)

        self.m_image_out_port.close_port()


class ImageStatisticsModule(ProcessingModule):
    """
    Pipeline module for calculating image statistics for the full images or a subsection of the
    images.
    """

    @typechecked
    def __init__(self,
                 name_in: str = 'im_stat',
                 image_in_tag: str = 'im_arr',
                 stat_out_tag: str = 'stat',
                 position: Union[Tuple[int, int, float], Tuple[None, None, float]] = None) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the images that are read as input.
        stat_out_tag : str
            Tag of the database entry with the statistical results that are written as output. The
            result is stored in the following order: minimum, maximum, sum, mean, median, and
            standard deviation.
        position : tuple(int, int, float)
            Position (x, y) (pix) and radius (arcsec) of the circular area in which the statistics
            are calculated. The full image is used if set to None.

        Returns
        -------
        NoneType
            None
        """

        super(ImageStatisticsModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_stat_out_port = self.add_output_port(stat_out_tag)

        self.m_position = position

    def run(self) -> None:
        """
        Run method of the module. Calculates the minimum, maximum, sum, mean, median, and standard
        deviation of the pixel values of each image separately. NaNs are ignored for each
        calculation. The values are calculated for either the full images or a circular
        subsection of the images.

        Returns
        -------
        NoneType
            None
        """

        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')

        nimages = self.m_image_in_port.get_shape()[0]
        im_shape = self.m_image_in_port.get_shape()[1:]

        if self.m_position is None:
            indices = None

        else:
            if self.m_position[0] is None and self.m_position[1] is None:
                center = center_pixel(self.m_image_in_port[0, ])

                self.m_position = (center[0], # y position
                                   center[1], # x position
                                   self.m_position[2]/pixscale) # radius (pix)

            else:
                self.m_position = (int(self.m_position[1]), # y position
                                   int(self.m_position[0]), # x position
                                   self.m_position[2]/pixscale) # radius (pix)

            rr_grid = pixel_distance(im_shape, self.m_position)
            rr_reshape = np.reshape(rr_grid, (rr_grid.shape[0]*rr_grid.shape[1]))
            indices = np.where(rr_reshape <= self.m_position[2])[0]

        def _image_stat(image_in, indices):
            if indices is None:
                image_select = np.copy(image_in)

            else:
                image_reshape = np.reshape(image_in, (image_in.shape[0]*image_in.shape[1]))
                image_select = image_reshape[indices]

            nmin = np.nanmin(image_select)
            nmax = np.nanmax(image_select)
            nsum = np.nansum(image_select)
            mean = np.nanmean(image_select)
            median = np.nanmedian(image_select)
            std = np.nanstd(image_select)

            return np.asarray([nmin, nmax, nsum, mean, median, std])

        self.apply_function_to_images(_image_stat,
                                      self.m_image_in_port,
                                      self.m_stat_out_port,
                                      'Running ImageStatisticsModule',
                                      func_args=(indices, ))

        history = f'number of images = {nimages}'
        self.m_stat_out_port.copy_attributes(self.m_image_in_port)
        self.m_stat_out_port.add_history('ImageStatisticsModule', history)
        self.m_stat_out_port.close_port()

class FrameSimilarityModule(ProcessingModule):
    """
    Pipeline module which measures the similarity frames using different techniques.
    """

    def __init__(self,
                 name_in="frame_comparison",
                 image_tag="im_arr",
                 method="MSE",
                 mask_radius=[0., 5.],
                 fwhm=.1,
                 temporal_median='slow'):
        """
        Constructor of FrameSimilarityModule

        source: https://iopscience.iop.org/article/10.3847/1538-3881/aafee2/pdf

        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_tag : str
            Tag of the database entry that is read as input.
        method : str
            Name of the similarity measure to be calculated
        mask_radius : list
            mask radii to be applied to the images, inner and outer.
        fwhm : float
            FWHM
        temporal_median : str
            option to calculate the temporal median every time('slow', like in source paper)
            or once for the entire set('fast', not like the source paper)

        Returns
        -------
        NoneType
            None
        """

        super(FrameSimilarityModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_tag)
        self.m_image_out_port = self.add_output_port(image_tag)

        assert method in ['MSE', 'PCC', 'SSIM'], "The chosen method '{}' is not"\
            " available. Please ensure that you have selected one of 'MSE', 'PCC', 'SSIM'"\
            .format(str(method))
        self.m_method = method

        assert temporal_median in ['slow', 'fast'], "The chosen temporal_median '{}' is"\
            " not available. Please ensure that you have selected one of 'slow', 'fast'"\
            .format(str(temporal_median))
        self.m_temporal_median = temporal_median

        self.m_mask_radii = mask_radius
        self.m_fwhm = fwhm

    @staticmethod
    def _similarity(images, reference_index, N_pix, mode, fwhm, temporal_median=False):
        """
        Internal function. Returns the MSE as defined by Ruane et al. 2019
        """
        def cov(P, Q):
            """
            Internal function. Returns the covariance as defined by Ruane et al. 2019
            """
            return 1 / (N_pix - 1) * np.sum((P - np.nanmean(P)) * (Q - np.nanmean(Q)))
        def std(P):
            """
            Internal function. Returns the standard deviation as defined by Ruane et al. 2019
            """
            return np.sqrt(1 / (N_pix - 1) * np.sum(((P - np.nanmean(P)))**2))

        def _temporal_median(reference_index, images):
            """
            Internal function. Calculates the temporal median for all frames, except the one \
                with the reference_index
            """
            M = np.concatenate((images[:reference_index], images[reference_index+1:]))
            return np.median(M, axis=0)

        X_i = images[reference_index]
        if not temporal_median:
            M = _temporal_median(reference_index, images=images)
        if mode == "MSE":
            return reference_index, compare_nrmse(X_i, M)

        elif mode == "PCC":
            PCC = cov(X_i, M) / (std(X_i) * std(M))
            del X_i, M
            return reference_index, PCC

        elif mode == "SSIM":
            if int(fwhm) % 2 == 0:
                winsize = int(fwhm) + 1
            else:
                winsize = int(fwhm)
            return reference_index, compare_ssim(X_i, M, win_size=winsize)

        # elif mode == "DSC":
        #     # make the images to binaries
        #     X_i -= np.median(X_i)
        #     X_i = X_i > 0
        #     M = np.mean(M)
        #     M = M > 0
        #     return reference_index, np.sum(2 * X_i * M / (np.sum(X_i) + np.sum(M)))


    def run(self):
        """
        Run method of the module. Compares individual frames to the others \
            using different techniques. Selects those which are most similar.

        Returns
        -------
        NoneType
            None
        """

        # get image number and image shapes
        nimages = self.m_image_in_port.get_shape()[0]
        im_shape = self.m_image_in_port.get_shape()[1:]

        # get pixscale
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        # convert arcsecs to pixels
        self.m_mask_radii = np.floor(np.array(self.m_mask_radii) / pixscale)
        self.m_fwhm = int(self.m_fwhm / pixscale)

        # overlay the same mask over all images
        mask = create_mask(im_shape, self.m_mask_radii)
        images = self.m_image_in_port.get_all()

        if self.m_temporal_median == 'fast':
            temporal_median = np.median(images, axis=0)
        else:
            temporal_median = False

        if self.m_method != 'SSIM':
            images *= mask
            
        # count mask pixels for normalization
        N_pix = int(np.sum(mask))
        # compare images and store similarity
        similarities = np.zeros(nimages)

        cpu = self._m_config_port.get_attribute("CPU")

        pool = mp.Pool(cpu)
        async_results = []

        start_time = time.time()
        for i in range(nimages):
            async_results.append(pool.apply_async(FrameSimilarityModule._similarity,
                                                  args=(images,
                                                        i,
                                                        N_pix,
                                                        self.m_method,
                                                        self.m_fwhm,
                                                        temporal_median)))

        pool.close()

        # wait for all processes to finish
        while mp.active_children():
            # number of finished processes
            nfinished = sum([i.ready() for i in async_results])

            progress(nfinished, nimages, 'Running FrameSimilarityModule', start_time)

            # check if new processes have finished every 5 seconds
            time.sleep(5)

        # get the results for every async_result object
        for async_result in async_results:
            reference, similarity = async_result.get()
            similarities[reference] = similarity

        pool.terminate()

        self.m_image_out_port.add_attribute("SIMILARITY" + "_" + self.m_method, \
            similarities, static=False)
        self.m_image_out_port.close_port()

class RemoveFramesByAttributeModule(ProcessingModule):
    """
    Pipeline module selects frames based on one of the image_in_tag attributes.
    """

    def __init__(self,
                 name_in="frame_selection",
                 image_in_tag="im_arr",
                 attribute_tag=None,
                 selected_frames=100,
                 order='descending',
                 selected_out_tag="im_arr_selected",
                 removed_out_tag="im_arr_removed"):
        """
        Constructor of RemoveFramesByAttributeModule

        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_tag : str
            Tag of the database entry that is read as input.
        attribute_tag : str
            Name of the attribute which is used to sort and select the frames
        selected_frames : int
            Number of frames which are selected.
        order : str
            Order in which the frames are selected are selected, can be either
            'descending' (will select the lowest attributes) or 'ascending' (
            will select the highest attributes)
        selected_out_tag : str
            Tag of the database entry to which the selected frames are written
        removed_out_tag : str
            Tag of the database entry to which the removed frames are written

        Returns
        -------
        NoneType
            None

        Example 1
        ---------
        The example below selects the first('ascending') 100 frames('INDEX')
        of a the frames saved in 'im_arr' and writes them to 'im_arr_selected'.

        RemoveFramesByAttributeModule(
            name_in="frame_selection",
            image_in_tag="im_arr",
            attribute_tag='INDEX',
            selected_frames=100,
            order='ascending',
            selected_out_tag="im_arr_selected",
            removed_out_tag="im_arr_removed"))

        Example 2
        ---------
        The example below selects the largest 200 SSIM valued frames('descending')
        200 frames('INDEX') of a the frames saved in 'im_arr' and writes them to
        'im_arr_selected'.

        RemoveFramesByAttributeModule(
            name_in="frame_selection",
            image_in_tag="im_arr",
            attribute_tag='SSIM',
            selected_frames=200,
            order='descending',
            selected_out_tag="im_arr_selected",
            removed_out_tag="im_arr_removed"))
        """
        super(RemoveFramesByAttributeModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_selected_out_port = self.add_output_port(selected_out_tag)
        self.m_removed_out_port = self.add_output_port(removed_out_tag)

        self.m_attribute_tag = attribute_tag
        self.m_frames = selected_frames

        assert order in ['ascending', 'descending'], 'The selected order is not available. The'\
            'available options are "ascending" or "descending"'
        self.m_order = order

    # copied from FrameSelectionModule
    def _initialize(self):
        if self.m_image_in_port.tag == self.m_selected_out_port.tag or \
                self.m_image_in_port.tag == self.m_removed_out_port.tag:
            raise ValueError("Input and output ports should have a different tag.")

        if self.m_selected_out_port is not None:
            self.m_selected_out_port.del_all_data()
            self.m_selected_out_port.del_all_attributes()
            print("removed all old data")

        if self.m_removed_out_port is not None:
            self.m_removed_out_port.del_all_data()
            self.m_removed_out_port.del_all_attributes()
            print("removed all old data")

    def run(self):
        """
        Run function of RemoveFramesByAttributeModule
        """
        self._initialize()
        images = self.m_image_in_port.get_all()
        nimages = images.shape[0]

        # grab the attribute
        if self.m_attribute_tag in ['MSE', 'PCC', 'SSIM', 'DSC']:
            attribute = self.m_image_in_port.get_attribute("SIMILARITY_{}"\
                .format(self.m_attribute_tag))
        else:
            attribute = self.m_image_in_port.get_attribute("{}"\
                .format(self.m_attribute_tag))

        index = self.m_image_in_port.get_attribute("INDEX")

        if self.m_order == 'descending':
            # sort attribute in descending order
            sorting_order = np.argsort(attribute)[::-1]
        else:
            # sort attribute in ascending order
            sorting_order = np.argsort(attribute)

        attribute = attribute[sorting_order]
        index = index[sorting_order]

        removed_index = index[:self.m_frames]
        selected_index = index[self.m_frames:]

        indices = removed_index

        # copied from FrameSelectionModule ... 
        # possibly refactor to @staticmethod or move to util.remove
        if np.size(indices) > 0:
            memory = self._m_config_port.get_attribute("MEMORY")
            frames = memory_frames(memory, nimages)

            if memory == 0 or memory >= nimages:
                memory = nimages

            for i, _ in enumerate(frames[:-1]):
                images = self.m_image_in_port[frames[i]:frames[i+1], ]

                index_del = np.where(np.logical_and(indices >= frames[i], \
                                                    indices < frames[i+1]))

                write_selected_data(
                    images,
                    indices[index_del]%memory,
                    self.m_removed_out_port,
                    self.m_selected_out_port)

        else:
            warnings.warn("No frames were removed.")


        if self.m_selected_out_port is not None:
            # Copy attributes before write_selected_attributes is used
            self.m_selected_out_port.copy_attributes(self.m_image_in_port)

        if self.m_removed_out_port is not None:
            # Copy attributes before write_selected_attributes is used
            self.m_removed_out_port.copy_attributes(self.m_image_in_port)

        # write the selected and removed data to the respective output ports
        write_selected_attributes(
            selected_index,
            self.m_image_in_port,
            self.m_removed_out_port,
            self.m_selected_out_port)
