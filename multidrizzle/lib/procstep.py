from __future__ import division # confidence high

import time

def printout(text):
    print(' *** ' + text + '\n ***')

def timestamp():
    print('\n----------------------------')
    print time.strftime('%c %Z', time.localtime(time.time()))
    print('----------------------------\n')

class ProcSteps:
    """ The ProcSteps class encapsulates the logic for deciding
        which processing steps get performed by Multidrizzle based
        on user or mdriztab input. It also keeps track of elapsed
        time used by each step.
    """

    # Step names are kept as class variables so they can be
    # used to access individual step info.

    doInitialize    = '1  Initialize:         '
    doBuildStatic   = '2  Build static mask:  '
    doSky           = '3  Subtract sky:       '
    doDrizSeparate  = '4  Drizzle separate:   '
    doMedian        = '5  Median/sum/ave.:    '
    doBlot          = '6  Blot:               '
    doDrizCR        = '7  DrizCR:             '
    doFinalDriz     = '8  Final drizzle:      '

    __report_header = '   Step                Elapsed time'

    def __init__(self):

        # Step objects are kept in a dictionary keyed by step names.
        # Dictionary starts with initialization step added in.

        self.__steps = {}

        self.__steps[ProcSteps.doInitialize] = \
            ProcStep(True, 'Initializing...')

    def addSteps(self, static, skysub, driz_separate, median, blot,
                 driz_cr, driz_combine):

        self.__steps[ProcSteps.doBuildStatic] = \
            ProcStep(static,'Building static bad-pixel mask...')

        self.__steps[ProcSteps.doSky] = \
            ProcStep(skysub,'Subtracting sky...')

        self.__steps[ProcSteps.doDrizSeparate] = \
            ProcStep(driz_separate,'Drizzling separate...')

        self.__steps[ProcSteps.doMedian] = \
            ProcStep(median,'Computing combined image...')

        self.__steps[ProcSteps.doBlot] = \
            ProcStep(blot,'Blotting back the combined image...')

        self.__steps[ProcSteps.doDrizCR] = \
                ProcStep(driz_cr,'Doing driz_cr...')

        self.__steps[ProcSteps.doFinalDriz] = \
            ProcStep(driz_combine,'Doing final drizzle...')

    def getFlag(self, step_name):
        """ Gets the boolean flag associated with this step. """

        return self.__getUserSelection(step_name)

    def doStep(self, step_name):
        """ Checks if a step should be performed. """

        if self.getFlag(step_name) and \
           self.__isCompleted(step_name) == False:

            self.__steps[step_name].recordStartTime()

            self.printTimestamp(step_name)

            return True

        else:
            return False

    def printTimestamp(self, step_name):
        """ Prints a time stamp message. """

        self.__steps[step_name].printTimestamp()

    def reportTimes(self):
        """ Generates report with elapsed times used by each step. """
        keys = self.__steps.keys()
        keys.sort()

        print '\n' + ProcSteps.__report_header + '\n'

        _total = 0
        for key in keys:
            _time = self.__steps[key].getElapsedTime()
            _total += _time
            print key + str(_time) + ' sec.'

        print '   Total               ' + str(_total) + 'sec.'

    # Methods delegate the query to the step object asociated
    # with the provided step name.

    def markStepDone(self, step_name):
        """ Records that step has been completed. """
        self.__steps[step_name].markStepDone()

    def resetStep(self, step_name):
        """ Resets the status of the processing step to not completed."""
        self.__steps[step_name].resetStep()

    def __getUserSelection(self, step_name):
        """ Returns status of user setting. """
        return self.__steps[step_name].getUserSelection()

    def __isCompleted(self, step_name):
        """ Returns status of step processing: complete or not. """
        return self.__steps[step_name].isCompleted()


class ProcStep:
    """ This class encapsulates the information
        associated with the processing status of
        a single execution step.
    """

    def __init__(self, user_switch, message):
        self.__message = message
        self.__switch = user_switch
        self.__elapsed_time = 0

        self.resetStep()

    def recordStartTime(self):
        self.__start_time = time.time()

    def markStepDone(self):
        self.__elapsed_time = time.time() - self.__start_time
        self.__completed = True

    def isCompleted(self):
        return self.__completed

    def resetStep(self):
        self.recordStartTime()
        self.__completed = False

    def getUserSelection(self):
        return self.__switch

    def getElapsedTime(self):
        return self.__elapsed_time

    def printTimestamp(self):
        timestamp()
        printout(self.__message)
        print('')
