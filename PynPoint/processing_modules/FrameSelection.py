"""
Modules with tools for frame selection.
"""

import sys
import math

import numpy as np

from scipy.ndimage.filters import gaussian_filter
from astropy.nddata import Cutout2D

from PynPoint.core import ProcessingModule
from PynPoint.util.Progress import progress


class RemoveFramesModule(ProcessingModule):
    """
    Module for removing frames.
    """

    def __init__(self,
                 frame_indices,
                 name_in="remove_frames",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_remove"):
        """
        Constructor of RemoveFramesModule.

        :param frame_indices: Frame indices to be removed. Python indexing starts at 0.
        :type frame_indices: tuple or array, int
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str

        :return: None
        """

        super(RemoveFramesModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_frame_indices = np.asarray(frame_indices)

    def run(self):
        """
        Run method of the module. Removes the frames, removes the associated NEW_PARA values,
        updates the NAXIS3 value, and saves the data and attributes.

        :return: None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        memory = self._m_config_port.get_attribute("MEMORY")

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise ValueError("Input and output port should have a different tag.")

        if np.size(np.where(self.m_frame_indices >= self.m_image_in_port.get_shape()[0])) > 0:
            raise ValueError("Some values in frame_indices are larger than the total number of "
                             "available frames, %s." % str(self.m_image_in_port.get_shape()[0]))

        if "NEW_PARA" not in self.m_image_in_port.get_all_non_static_attributes():
            raise ValueError("NEW_PARA not found in header. Parallactic angles should be "
                             "provided for all frames before any frames can be removed.")

        nframes = self.m_image_in_port.get_shape()[0]
        nstacks = int(float(nframes)/float(memory))

        # Reading subsets of 'memory' frames and removes frame_indices

        for i in range(nstacks):
            progress(i, nstacks, "Running RemoveFramesModule...")

            tmp_im = self.m_image_in_port[i*memory:(i+1)*memory, ]

            index_del = np.where(np.logical_and(self.m_frame_indices >= i*memory, \
                                 self.m_frame_indices < (i+1)*memory))

            if np.size(index_del) > 0:
                tmp_im = np.delete(tmp_im,
                                   self.m_frame_indices[index_del]%memory,
                                   axis=0)

            self.m_image_out_port.append(tmp_im)

        sys.stdout.write("Running RemoveFramesModule... [DONE]\n")
        sys.stdout.flush()

        # Adding the leftover frames that do not fit in an integer amount of 'memory'

        index_del = np.where(self.m_frame_indices >= nstacks*memory)[0]

        if np.size(index_del) > 0:
            tmp_im = self.m_image_in_port[nstacks*memory: \
                                          self.m_image_in_port.get_shape()[0], ]

            tmp_im = np.delete(tmp_im,
                               self.m_frame_indices[index_del]%memory,
                               axis=0)

            self.m_image_out_port.append(tmp_im)

        # Attributes

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        parang = self.m_image_in_port.get_attribute("NEW_PARA")

        self.m_image_out_port.add_attribute("NEW_PARA",
                                            np.delete(parang, self.m_frame_indices),
                                            static=False)

        if "STAR_POSITION" in self.m_image_in_port.get_all_non_static_attributes():

            position = self.m_image_in_port.get_attribute("STAR_POSITION")
            self.m_image_out_port.add_attribute("STAR_POSITION",
                                                np.delete(position, self.m_frame_indices, axis=0),
                                                static=False)

        nframes_in = self.m_image_in_port.get_attribute("NFRAMES")
        nframes_out = np.copy(nframes_in)

        total = 0
        for i, frames in enumerate(nframes_in):
            index_del = np.where(np.logical_and(self.m_frame_indices >= total, \
                                 self.m_frame_indices < total+frames))[0]

            nframes_out[i] -= np.size(index_del)

            total += frames

        self.m_image_out_port.add_attribute("NFRAMES", nframes_out, static=False)

        self.m_image_out_port.add_history_information("Removed frames",
                                                      str(np.size(self.m_frame_indices)))

        self.m_image_in_port.close_port()


class FrameSelectionModule(ProcessingModule):
    """
    Module for frame selection.
    """

    def __init__(self,
                 name_in="frame_selection",
                 image_in_tag="im_arr",
                 selected_out_tag="im_arr_selected",
                 removed_out_tag="im_arr_removed",
                 fwhm=0.2,
                 threshold=3.):
        """
        Constructor of FrameSelectionModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param selected_out_tag: Tag of the database entry with the selected images that are
                                 written as output. Should be different from *image_in_tag*.
        :type selected_out_tag: str
        :param removed_out_tag: Tag of the database entry with the removed images that are
                                written as output. Should be different from *image_in_tag*.
        :type removed_out_tag: str
        :param fwhm: The full width at half maximum (FWHM) of the Gaussian kernel (arcsec) that is
                     used to smooth the images before the brightest pixel is located. Should be
                     similar in size to the FWHM of the stellar PSF. The *fwhm* is also used as
                     aperture radius for measuring the photometry around the location of the
                     brightest pixel.
        :type fwhm: float
        :param threshold: Threshold in units of sigma for the frame selection. All images that
                          are a *threshold* number of sigmas away from the median photometry will
                          be removed.
        :type threshold: float

        :return: None
        """

        super(FrameSelectionModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_selected_out_port = self.add_output_port(selected_out_tag)
        self.m_removed_out_port = self.add_output_port(removed_out_tag)

        self.m_fwhm = fwhm
        self.m_threshold = threshold

    def run(self):
        """
        Run method of the module. Smooths the images with a Gaussian kernel, locates the brightest
        pixel in each image, measures the integrated flux around the brightest pixel, calculates
        the median and standard deviation of the photometry, and applies sigma clipping to remove
        images that are of poor quality (e.g., due to opening of the AO loop).

        :return: None
        """

        if self.m_image_in_port.tag == self.m_selected_out_port.tag or \
                self.m_image_in_port.tag == self.m_removed_out_port.tag:
            raise ValueError("Input and output ports should have a different tag.")

        if "NEW_PARA" not in self.m_image_in_port.get_all_non_static_attributes():
            raise ValueError("NEW_PARA not found in header. Parallactic angles should be "
                             "provided for all frames before a frame selection can be applied.")

        self.m_selected_out_port.del_all_data()
        self.m_selected_out_port.del_all_attributes()

        self.m_removed_out_port.del_all_data()
        self.m_removed_out_port.del_all_attributes()

        memory = self._m_config_port.get_attribute("MEMORY")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")
        parang = self.m_image_in_port.get_attribute("NEW_PARA")

        self.m_fwhm /= pixscale
        gaussian_sigma = self.m_fwhm/math.sqrt(8.*math.log(2.))

        nframes = self.m_image_in_port.get_shape()[0]
        nstacks = int(float(nframes)/float(memory))

        position = np.zeros((nframes, 2), dtype=np.int64)
        phot = np.zeros(nframes)

        for i in range(nframes):
            progress(i, nframes+nframes, "Running FrameSelectionModule...")

            im_smooth = gaussian_filter(self.m_image_in_port[i],
                                        gaussian_sigma,
                                        truncate=4.)

            position[i, :] = np.unravel_index(im_smooth.argmax(), im_smooth.shape)

            im_cut = Cutout2D(im_smooth, (position[i, 1], position[i, 0]), size=2.*self.m_fwhm).data

            if i == 0:
                npix = im_cut.shape[0]

                if npix%2 == 0:
                    x_grid = y_grid = np.linspace(-npix/2+0.5, npix/2-0.5, npix)
                elif npix%2 == 1:
                    x_grid = y_grid = np.linspace(-(npix-1)/2, (npix-1)/2, npix)

                xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
                rr_grid = np.sqrt(xx_grid*xx_grid+yy_grid*yy_grid)

            im_cut[rr_grid > self.m_fwhm] = 0.

            phot[i] = np.sum(im_cut)

        phot_med = np.median(phot)
        phot_std = np.std(phot)

        index_rm = np.logical_or((phot > phot_med+self.m_threshold*phot_std),
                                 (phot < phot_med-self.m_threshold*phot_std))

        for i in range(nstacks):
            progress(nframes+i*nframes/nstacks, nframes+nframes, "Running FrameSelectionModule...")

            index = index_rm[i*memory:i*memory+memory, ]
            image = self.m_image_in_port[i*memory:i*memory+memory, ]

            self.m_selected_out_port.append(image[np.logical_not(index)])
            self.m_removed_out_port.append(image[index])

        sys.stdout.write("Running FrameSelectionModule... [DONE]\n")
        sys.stdout.flush()

        self.m_selected_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_removed_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        self.m_selected_out_port.add_attribute("NEW_PARA",
                                               parang[np.logical_not(index_rm)],
                                               static=False)

        self.m_removed_out_port.add_attribute("NEW_PARA",
                                              parang[index_rm],
                                              static=False)

        self.m_selected_out_port.add_attribute("STAR_POSITION",
                                               position[np.logical_not(index_rm)],
                                               static=False)

        self.m_removed_out_port.add_attribute("STAR_POSITION",
                                              position[index_rm],
                                              static=False)

        nframes_in = self.m_image_in_port.get_attribute("NFRAMES")

        nframes_del = np.zeros(np.size(nframes_in), dtype=np.int64)
        nframes_sel = np.zeros(np.size(nframes_in), dtype=np.int64)

        total = 0
        for i, frames in enumerate(nframes_in):
            nframes_del[i] = np.size(np.where(index_rm[total:total+frames])[0])
            nframes_sel[i] = frames - nframes_del[i]
            total += frames

        self.m_selected_out_port.add_attribute("NFRAMES", nframes_sel, static=False)
        self.m_removed_out_port.add_attribute("NFRAMES", nframes_del, static=False)

        n_rm = np.size(index_rm[index_rm])

        self.m_selected_out_port.add_history_information("Frame selection",
                                                         str(n_rm)+" images removed")

        self.m_removed_out_port.add_history_information("Frame selection",
                                                        str(n_rm)+" images removed")

        self.m_image_in_port.close_port()