"""
This example shows the Python part for an experiment with backscatter-triggered induction &
harvest after a fixed time followed by an assay using the reader.

It uses nine EVOware scripts, that the user has to provide. They shall be named like this:
    1. Pahpshmir_Preparing.esc -- e.g. for loading TARA into the centrifuge
    2. Pahpshmir_Induction.esc -- e.g. for inducing some columns of the FlowerPlate using a .gwl
    3. Pahpshmir_Sampling.esc -- e.g. for transferring culture from from the FP to sampling plate using a .gwl 
    4. Pahpshmir_Processing.esc -- e.g. for centrifugation of sampling plate and transfer to supernatant plate using a .gwl
    5. Pahpshmir_Finishing.esc -- e.g. for unloading centrifuge and set T to 30 degrees
For the sGFP assay using the reader:
    6. Pahpshmir_ReaderLoad.esc -- e.g. for preparing an assay and loading the reader with the measurement plate
    7. Pahpshmir_ReaderMeasure.esc -- e.g. for the reader measurement 
    8. Pahpshmir_ReaderShake.esc -- to shake plate inside reader between measurements
    9. Pahpshmir_ReaderUnload.esc -- e.g. to unload the reader after the measurement
For the cutinase assay using the reader:
    10. Pahpshmir_CutinaseAssay.esc
    11. Pahpshmir_CutinaseAssay_Standard.esc
    12. Pahpshmir_Reader37.esc

Furthermore, a BioLection file 'CM_Cglutamicum' is needed for cultivation. 

Starting/Pausing/Stopping the BioLector is handled from the Python script.
"""

# import standard libraries
import logging
import pathlib
import time
import numpy
import pandas

# import custom packages, such as dibecs_contrib, sila_biolector, sila_evoware...
import bletl
import bletl_analysis
import bletl_pro
import dibecs
import dibecs_contrib
import sila_biolector
import sila_evoware

from robotools import liquidhandling
from robotools import evotools

# by using a named logger, the logging events can be associated with this script
logger = logging.getLogger('main')

#TODO: Check for True and False
biolector = False
execute_cutinase = False
execute_sGFP = True


class Experiment(dibecs.Continuous):
    def __init__(self):
        self.cultivator = dibecs_contrib.Cultivator()
        self.eventlog = dibecs_contrib.utils.EventLog('eventlog.xlsx')

        # initialize labware objects
        # initial FP volume = 1000, because the sampling step pipettes 950 to make sure that the well is emty, max volume = 1200, because initial volume + induction volume = 1100
        self.flowerplate = liquidhandling.Labware('Cultivation', 6, 8, min_volume=0, max_volume=1200, initial_volumes=1000)
        self.inducing = liquidhandling.Labware('IPTG', 6, 4, min_volume=50, max_volume=1500, initial_volumes=1200)
        self.sampling = liquidhandling.Labware('SampleDWP', 8, 12, min_volume=0, max_volume=1000)
        self.supernatants = liquidhandling.Labware('SupernatantDWP', 8, 12, min_volume=100, max_volume=1000)     

        # initialize well rotator to map flowerplate to the left half of the 96 DWP
        self.sampling_rotator = dibecs_contrib.transform.WellRotator(original_shape=self.flowerplate.wells.shape)

        self.robot = None
        self.biolector = None

        # Triggering
        # Three kinds of wells: active_wells, induced_wells and sampled_wells
        self.active_wells = [
            f'{r}{c:02d}'
            for r in 'ABCDEF'
            for c in range(1, 8+1)
        ]

        # induced_wells is a dictionary to do 'well in self.induced_wells or `harvesttime = self.induced_wells[well]`
        self.induced_wells = {}

        self.sampled_wells = []
        super().__init__()

    def setup(self):

        logger.info('Connecting to Evoware service')
        self.robot = sila_evoware.PyEvowareClient(self.info.channels[f'{self.info.platform.name}-Evoware'])

        if biolector is True:
            logger.info('Connecting to BioLector')
            self.biolector = sila_biolector.PyBiolectorClient(self.info.channels[f'{self.info.platform.name}-BioLector'])

        return

    def pre_loop(self):

        if biolector is True:
        
            # Prepare cultivation, start BioLection file
            self.slack.info('Starting BioLector...')
            self.cultivator.run_cultivation(self.biolector, 'CM_Cglutamicum')
            self.slack.info('Pre-Loop', f'The cultivation has started at {self.cultivator.start_utc}.')

            # Wash tips and load centrifuge
            self.robot.execute_protocol_async(f'{self.info.platform.name}_Preparing.esc').wait()
            self.slack.info('Pre-Loop', 'The robot was prepared. Centrifuge is loaded.')
        logger.info('Starting experiment')

        return

    def select_for_induction(self, latest_cycle:int, bldata:bletl.BLData) -> list:
        """Method for making induction decisions.

        Args:
            latest_cycle (int): number of the current cycle (used to slice data)
            bldata (bletl.BLData): current BioLector data for making decisions

        Returns:
            wells_to_induce (list): list of wells that were triggered in this cycle
        """
        # The -> list: defines that a list is returned 

        wells_to_induce = []
        for well in self.active_wells:
            # get backscatter data for this well, x_BS = time, y_BS = BS data
            x_BS, y_BS = bldata['BS3'].get_timeseries(well)
            x_DO, y_DO = bldata['DO'].get_timeseries(well)

            x_BS = x_BS[:latest_cycle]
            y_BS = y_BS[:latest_cycle]

            x_DO = x_DO[:latest_cycle]
            y_DO = y_DO[:latest_cycle]

            # BS trigger:
            if y_BS[latest_cycle - 1] >= 5.82 and not well in self.induced_wells:
                wells_to_induce.append(well)

            # Induction without trigger, if it takes too long!
            if self.cultivator.cultivation_time > 16 and not well in self.induced_wells:
                wells_to_induce.append(well)    

        return wells_to_induce

    def induce_wells(self, wells:list, volumes:float):
        """ Method for inducing wells in BioLector 

        Args:
            wells: well-IDs of induced wells
            volumes: induction volume
        Returns:
            nothing
        """

        if len(wells) == 0:
            return

        # reformat the parameters into arrays of the same size, one induction volume is given, if more than one well should be induced, the volume applies to each of these wells
        wells = numpy.atleast_1d(wells)
        volumes = numpy.atleast_1d(volumes)
        if len(volumes) == 1:
            volumes = numpy.repeat(volumes, len(wells))

        self.slack.info('Induction', f'Inducing wells {wells}...')

        # write gwl for inducing wells in biolector
        with evotools.Worklist('Induction.gwl') as wl:
            wl.transfer(
                self.inducing, 'A01',
                self.flowerplate, wells,
                volumes,
                partition_by='destination' # With this kwarg pipetting steps are grouped by FP columns, which is faster than default (grouped by 'source')
            )

        # pause biolector and open cover
        self.biolector.pause_experiment_async(shaker_on=True, close_cover=False).wait()

        # perform induction
        self.robot.execute_protocol_async(f'{self.info.platform.name}_Induction.esc', additional_files=['Induction.gwl']).wait()

        # asynchronously resume measurements and close cover
        self.biolector.resume_experiment_async().wait()
        self.biolector.close_cover_async().wait()

        logger.info('New induced wells:\n')
        logger.info(self.inducing)
        return

    def select_for_sacrifice(self, latest_cycle:int, bldata:bletl.BLData) -> list:
        """Method for making sacrifice decisions.

        Args:
            latest_cycle (int): number of the current cycle (used to slice data)
            bldata (bletl.BLData): current BioLector data for making decisions

        Returns:
            wells_to_sacrifice (list): list of wells that were triggered in this cycle
        """

        wells_to_sacrifice = []
        for well, harvesttime in self.induced_wells.items():
                   
            if self.cultivator.cultivation_time >= harvesttime and not well in self.sampled_wells:
                wells_to_sacrifice.append(well)

        return wells_to_sacrifice

    def sample_wells(self, wells:list, volumes:float, latest_cycle): 
        """ Method for sampling from BioLector 

        Args:
            wells: well-IDs of sampled wells
            volumes: sampling volume
        Returns:
            nothing
        """

        if len(wells) == 0:
            return

        # reformat the parameters into arrays of the same size, one sampling volume is given, but if more than one well should be sampled, the volume applies to each of these wells
        wells = numpy.atleast_1d(wells)
        volumes = numpy.atleast_1d(volumes)
        if len(volumes) == 1:
            volumes = numpy.repeat(volumes, len(wells))

        self.slack.info('Sampling', f'Sampling from wells {wells}...')

        # write gwl for sampling from biolector
        with evotools.Worklist('Sampling.gwl') as wl:
            wl.transfer(
                self.flowerplate, wells,
                self.sampling, self.sampling_rotator.rotate_ccw(wells),
                volumes
            )

        # write gwl for transferring supernatants to supernatant DWP
        with evotools.Worklist('Processing.gwl') as wl:
            wl.transfer(
                self.sampling, self.sampling_rotator.rotate_ccw(wells),
                self.supernatants, self.sampling_rotator.rotate_ccw(wells),
                volumes,
                liquid_class='Supernatant_AspZmax-5'  # use special liquid class to pipet from above the pellet
            )

        # pause biolector and open cover
        self.biolector.pause_experiment_async(shaker_on=True, close_cover=False).wait()

        # make notes
        for well, volume in zip(wells, volumes):
            self.eventlog.write('samplings', time=self.cultivator.cultivation_time, well=well, cycle=latest_cycle, volume=-volume)

        # perform sampling
        self.robot.execute_protocol_async(f'{self.info.platform.name}_Sampling.esc', additional_files=['Sampling.gwl']).wait()

        # asynchronously resume measurements and close cover
        t_blresume = self.biolector.resume_experiment_async()

        # perform Pahpshmir_Processing
        self.robot.execute_protocol_async(f'{self.info.platform.name}_Processing.esc', additional_files=['Processing.gwl']).wait()

        # wait for the completion of the BioLector resume command
        # (it already finished at this point, but this is still important to catch errors)
        t_blresume.wait()
        self.biolector.close_cover_async().wait()

        logger.info('New supernatants:\n')
        logger.info(self.supernatants)
        return

    def loop(self):

        if biolector is True:

            latest_cycle = self.biolector.next_cycle_async().wait()
            logger.info(f'Downloading result file of cycle {latest_cycle}')
            current_filename = 'BioLectorData.csv'
            self.biolector.get_result_file_async(self.biolector.active_experiment.result_file_name, save_path=current_filename).wait()

            logger.info('Reading online data...')
            bldata = bletl_pro.parse(current_filename)

            # make decisions from BioLector data
            # induce, if BS correlates to 4 g/l CDW
            wells_to_induce = self.select_for_induction(latest_cycle, bldata)

            self.induce_wells(wells_to_induce, volumes=10)
            for well in wells_to_induce:
                harvesttime = self.cultivator.cultivation_time + 4.0
                self.eventlog.write('inductions', time=self.cultivator.cultivation_time, well=well, cycle=latest_cycle, volume=10, harvesttime=harvesttime)
                self.induced_wells[well] = harvesttime

            # harvest, if induction was 4 h ago
            wells_to_sample = self.select_for_sacrifice(latest_cycle, bldata)
            self.sample_wells(wells_to_sample, volumes=950, latest_cycle=latest_cycle)

            for well in wells_to_sample:
                self.active_wells.remove(well)
                self.sampled_wells.append(well)

            if len(self.active_wells) == 0:
                self.slack.info('Break-Loop', 'No active cultivation wells are left.')
                self.break_loop()

            # # This might be unnecessary, all wells are induced after 14 h and then harvsted 4 h later!
            # if self.cultivator.cultivation_time > 24:
            #     self.slack.info('Break-Loop', f'Sampling remaining wells: {self.active_wells}')
            #     self.sample_wells(self.active_wells, volumes=950, latest_cycle=latest_cycle)
            #     for well in self.active_wells:
            #         self.active_wells.remove(well)
            #         self.sampled_wells.append(well)
            #     self.slack.info('Break-Loop', 'Sampling complete. Stopping the loop.')
            #     self.break_loop()
        if biolector is False:
            self.break_loop()

        return

    def post_loop(self):

        if biolector is True:
            self.slack.info('Post-Loop', 'Washing tips and unloading centrifuge...') # Remove Tara plate from centrifuge after every well has been harvested
            t_finishing = self.robot.execute_protocol_async(f'{self.info.platform.name}_Finishing.esc')

            self.slack.info('Post-Loop', f'Stopping cultivation after {self.cultivator.cultivation_time:.2f} h. Downloading BioLector results...')
            filename = self.biolector.stop_experiment_async().wait()
            self.biolector.get_result_file_async(filename).wait()

            t_finishing.wait()
            self.slack.info('Post-Loop', 'The centrifuge was unloaded.')
            self.slack.info('Post-Loop', 'The cultivation completed successfully.')

        if execute_cutinase is True:
            logger.info('Starting cutinase assay')
            self.run_cutinaseassay()

        if execute_sGFP is True:
            logger.info('Starting sGFP assay')
            self.run_sGFPassay()

        logger.info('Finishing experiment')
        return

    def run_cutinaseassay(self):

        self.slack.info('Cutinase assay', 'Waiting for human to prepare cutinase assay (and sGFP assay)...')
        cutinase_response = self.rodi.ask_question('Do you want to proceed with cutinase assay?', ['Yes', 'No']).wait()

        if cutinase_response == 'Yes':
            self.slack.info('Cutinase assay', 'Starting cutinase assay...')
            self.robot.execute_protocol_async(f'{self.info.platform.name}_CutinaseAssay.esc', additional_files=[
                f'{self.info.platform.name}_CutinaseAssay_Standard.esc',
                f'{self.info.platform.name}_Reader37.esc'
            ]).wait()
            logger.info('Cutinase assay completed')

        else:
            logger.info('Cutinase assay was aborted via RODI.')

        self.slack.info('Cutinase assay', 'Cutinase assay completed.')
        return

    def run_sGFPassay(self):

#        logger.info('Loading reader...')
#        self.slack.info('sGFP assay', 'Preparing assay...')
#        self.robot.execute_protocol_async(f'{self.info.platform.name}_ReaderLoad.esc').wait()

#        self.slack.info('sGFP assay', 'Waiting for human to proceed...')
        sGFP_response = self.rodi.ask_question('Do you want to proceed with sGFP assay?', ['Yes', 'No']).wait()

        if sGFP_response == 'Yes':

            self.slack.info('sGFP assay', 'Measurement start')
            t_start = time.time()
            i = 0

            stop_question = self.rodi.ask_question('Do you want to abort the sGFP assay?', ['Yes'])

            while time.time() - t_start < 17*3600: #TODO: Anpassen an Robotikbuchung, mind 17 für 16 h!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                logger.info('Measurement')
                i += 1
                self.robot.execute_protocol_async(f'{self.info.platform.name}_ReaderMeasure.esc', result_dir=f'read_{i:02d}').wait()

                if stop_question.response:
                    logger.info('sGFP assay was aborted via RODI.')
                    break

                self.robot.execute_protocol_async(f'{self.info.platform.name}_ReaderShake.esc', result_dir='Trash').wait()    

        else:
            logger.info('sGFP assay was aborted via RODI before start of first measurement.')

        self.slack.info('sGFP assay', 'Unloading reader...')
        self.robot.execute_protocol_async(f'{self.info.platform.name}_ReaderUnload.esc').wait()
        self.slack.info('sGFP assay', 'Assay completed.')
        logger.info('sGFP assay completed')
        return


if __name__ == '__main__':
    exp = Experiment()
    result = exp()
