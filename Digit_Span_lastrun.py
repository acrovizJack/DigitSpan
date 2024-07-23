#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.4),
    on July 23, 2024, at 15:33
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.4'
expName = 'Digit_Span'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    '姓名': '王小明',
    '電話': '0900-000-000',
    '電子郵件': 'YourMail@gmail.com',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1080, 1920]
_loggingLevel = logging.getLevel('warning')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['姓名'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\fuchi\\Documents\\PsychoPy\\DigitSpan\\Digit_Span_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=1,
            winType='pyglet', allowStencil=True,
            monitor='ExperimentMonitor', color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='fill',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [1.0000, 1.0000, 1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'fill'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('EndIntroKey') is None:
        # initialise EndIntroKey
        EndIntroKey = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='EndIntroKey',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('endKey') is None:
        # initialise endKey
        endKey = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='endKey',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "intro" ---
    movie = visual.MovieStim(
        win, name='movie',
        filename='stimuli/opening.mp4', movieLib='ffpyplayer',
        loop=True, volume=1.0, noAudio=False,
        pos=(0, 0), size=(1.789, 1), units=win.units,
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=0
    )
    EndIntroKey = keyboard.Keyboard(deviceName='EndIntroKey')
    mouse_4 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_4.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "Instruction2" ---
    # Run 'Begin Experiment' code from code_5
    slideN = 1
    instruct_txt = '在這個實驗中，\n\n你需要嘗試記住螢幕上顯示的數字。\n\n所有數字都在0到9之間。\n\n你會看到一串數字，依序顯示\n\n請記住整串數字'
    
    maxSlideN = 2
    minSlideN = 1
    digitSpan = 2
    backimg_2 = visual.ImageStim(
        win=win,
        name='backimg_2', 
        image='stimuli/redesign/digitspan_background.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(1.775,1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    i2_txt = visual.TextStim(win=win, name='i2_txt',
        text='',
        font='Microsoft JhengHei',
        pos=(0, 0.055), height=0.05, wrapWidth=None, ori=0.0, 
        color=[-0.6157, -0.6706, -0.0196], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    pgnum = visual.TextStim(win=win, name='pgnum',
        text='',
        font='Arial',
        pos=(0, -0.3), height=0.05, wrapWidth=None, ori=0.0, 
        color=[-0.6157, -0.6706, -0.0196], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    backButton_2 = visual.ImageStim(
        win=win,
        name='backButton_2', 
        image='stimuli/backButtonImage.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.25, -0.3), size=(0.185,0.06),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    nextButton_2 = visual.ImageStim(
        win=win,
        name='nextButton_2', 
        image='stimuli/nextButtonImage.png', mask=None, anchor='center',
        ori=0.0, pos=(0.25, -0.3), size=(0.185,0.06),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    start_2 = visual.ImageStim(
        win=win,
        name='start_2', 
        image='stimuli/startButtonImage.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.375), size=(0.185,0.06),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    mouse_3 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_3.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "reset_correct" ---
    
    # --- Initialize components for Routine "set_item" ---
    
    # --- Initialize components for Routine "Digit_Presentation" ---
    backimg_3 = visual.ImageStim(
        win=win,
        name='backimg_3', 
        image='stimuli/redesign/digitspan_background.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(1.775,1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='stimuli/redesign/plain.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    Fixation = visual.TextStim(win=win, name='Fixation',
        text='+',
        font='Arial',
        pos=(0, 0), height=0.15, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    pres_text = visual.TextStim(win=win, name='pres_text',
        text='',
        font='Arial',
        pos=(0, 0.005), height=0.15, wrapWidth=None, ori=0.0, 
        color=[-0.6078, -0.6706, -0.0118], colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "Recall" ---
    backimg_4 = visual.ImageStim(
        win=win,
        name='backimg_4', 
        image='stimuli/redesign/digitspan_background.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(1.775,1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    recall_txt = visual.TextStim(win=win, name='recall_txt',
        text='請試著回想剛剛出現的數字',
        font='Arial',
        pos=(0, 0.25), height=0.05, wrapWidth=None, ori=0.0, 
        color=[-0.6157, -0.6706, -0.0196], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    textbox = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         pos=(0, 0),     letterHeight=0.15,
         size=(1, 0.2), borderWidth=2.0,
         color=[-0.6078, -0.6706, -0.0118], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=[-1.0000, -1.0000, -1.0000],
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textbox',
         depth=-3, autoLog=True,
    )
    continue_button = visual.ImageStim(
        win=win,
        name='continue_button', 
        image='stimuli/redesign/continue_button.png', mask=None, anchor='center',
        ori=0.0, pos=(0.2, -0.2), size=(0.35,0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    clear_button = visual.ImageStim(
        win=win,
        name='clear_button', 
        image='stimuli/redesign/clear.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.2, -0.2), size=(0.35,0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "Feedback" ---
    backimg_5 = visual.ImageStim(
        win=win,
        name='backimg_5', 
        image='stimuli/redesign/digitspan_background.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(1.775,1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    feedback_txt = visual.TextStim(win=win, name='feedback_txt',
        text='',
        font='Arial',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color=[-0.6157, -0.6706, -0.0196], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "End" ---
    backimg_6 = visual.ImageStim(
        win=win,
        name='backimg_6', 
        image='stimuli/redesign/digitspan_background.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(1.775,1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    Thank_you = visual.TextStim(win=win, name='Thank_you',
        text='',
        font='Arial',
        pos=(0, 0), height=0.075, wrapWidth=None, ori=0.0, 
        color=[-0.6078, -0.6706, -0.0118], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    endImg = visual.ImageStim(
        win=win,
        name='endImg', 
        image='stimuli/redesign/end_experiment.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.2), size=(0.315,0.074),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    mouse_2 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_2.mouseClock = core.Clock()
    endKey = keyboard.Keyboard(deviceName='endKey')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "intro" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro.started', globalClock.getTime(format='float'))
    EndIntroKey.keys = []
    EndIntroKey.rt = []
    _EndIntroKey_allKeys = []
    # setup some python lists for storing info about the mouse_4
    mouse_4.x = []
    mouse_4.y = []
    mouse_4.leftButton = []
    mouse_4.midButton = []
    mouse_4.rightButton = []
    mouse_4.time = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    introComponents = [movie, EndIntroKey, mouse_4]
    for thisComponent in introComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *movie* updates
        
        # if movie is starting this frame...
        if movie.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            movie.frameNStart = frameN  # exact frame index
            movie.tStart = t  # local t and not account for scr refresh
            movie.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(movie, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'movie.started')
            # update status
            movie.status = STARTED
            movie.setAutoDraw(True)
            movie.play()
        
        # *EndIntroKey* updates
        waitOnFlip = False
        
        # if EndIntroKey is starting this frame...
        if EndIntroKey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            EndIntroKey.frameNStart = frameN  # exact frame index
            EndIntroKey.tStart = t  # local t and not account for scr refresh
            EndIntroKey.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(EndIntroKey, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'EndIntroKey.started')
            # update status
            EndIntroKey.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(EndIntroKey.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(EndIntroKey.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if EndIntroKey.status == STARTED and not waitOnFlip:
            theseKeys = EndIntroKey.getKeys(keyList=['left','right','space','return'], ignoreKeys=["escape"], waitRelease=False)
            _EndIntroKey_allKeys.extend(theseKeys)
            if len(_EndIntroKey_allKeys):
                EndIntroKey.keys = _EndIntroKey_allKeys[-1].name  # just the last key pressed
                EndIntroKey.rt = _EndIntroKey_allKeys[-1].rt
                EndIntroKey.duration = _EndIntroKey_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        # *mouse_4* updates
        
        # if mouse_4 is starting this frame...
        if mouse_4.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_4.frameNStart = frameN  # exact frame index
            mouse_4.tStart = t  # local t and not account for scr refresh
            mouse_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_4.started', t)
            # update status
            mouse_4.status = STARTED
            mouse_4.mouseClock.reset()
            prevButtonState = mouse_4.getPressed()  # if button is down already this ISN'T a new click
        if mouse_4.status == STARTED:  # only update if started and not finished!
            buttons = mouse_4.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    x, y = mouse_4.getPos()
                    mouse_4.x.append(x)
                    mouse_4.y.append(y)
                    buttons = mouse_4.getPressed()
                    mouse_4.leftButton.append(buttons[0])
                    mouse_4.midButton.append(buttons[1])
                    mouse_4.rightButton.append(buttons[2])
                    mouse_4.time.append(mouse_4.mouseClock.getTime())
                    
                    continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in introComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro" ---
    for thisComponent in introComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro.stopped', globalClock.getTime(format='float'))
    movie.stop()  # ensure movie has stopped at end of Routine
    # check responses
    if EndIntroKey.keys in ['', [], None]:  # No response was made
        EndIntroKey.keys = None
    thisExp.addData('EndIntroKey.keys',EndIntroKey.keys)
    if EndIntroKey.keys != None:  # we had a response
        thisExp.addData('EndIntroKey.rt', EndIntroKey.rt)
        thisExp.addData('EndIntroKey.duration', EndIntroKey.duration)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_4.x', mouse_4.x)
    thisExp.addData('mouse_4.y', mouse_4.y)
    thisExp.addData('mouse_4.leftButton', mouse_4.leftButton)
    thisExp.addData('mouse_4.midButton', mouse_4.midButton)
    thisExp.addData('mouse_4.rightButton', mouse_4.rightButton)
    thisExp.addData('mouse_4.time', mouse_4.time)
    thisExp.nextEntry()
    # the Routine "intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    Instruct_Loop = data.TrialHandler(nReps=500.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='Instruct_Loop')
    thisExp.addLoop(Instruct_Loop)  # add the loop to the experiment
    thisInstruct_Loop = Instruct_Loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisInstruct_Loop.rgb)
    if thisInstruct_Loop != None:
        for paramName in thisInstruct_Loop:
            globals()[paramName] = thisInstruct_Loop[paramName]
    
    for thisInstruct_Loop in Instruct_Loop:
        currentLoop = Instruct_Loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisInstruct_Loop.rgb)
        if thisInstruct_Loop != None:
            for paramName in thisInstruct_Loop:
                globals()[paramName] = thisInstruct_Loop[paramName]
        
        # --- Prepare to start Routine "Instruction2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Instruction2.started', globalClock.getTime(format='float'))
        i2_txt.setText(instruct_txt)
        pgnum.setText(str(slideN) + '/' + str(maxSlideN))
        key_resp_2.keys = []
        key_resp_2.rt = []
        _key_resp_2_allKeys = []
        # setup some python lists for storing info about the mouse_3
        mouse_3.x = []
        mouse_3.y = []
        mouse_3.leftButton = []
        mouse_3.midButton = []
        mouse_3.rightButton = []
        mouse_3.time = []
        mouse_3.clicked_name = []
        gotValidClick = False  # until a click is received
        # keep track of which components have finished
        Instruction2Components = [backimg_2, i2_txt, pgnum, key_resp_2, backButton_2, nextButton_2, start_2, mouse_3]
        for thisComponent in Instruction2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Instruction2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *backimg_2* updates
            
            # if backimg_2 is starting this frame...
            if backimg_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                backimg_2.frameNStart = frameN  # exact frame index
                backimg_2.tStart = t  # local t and not account for scr refresh
                backimg_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(backimg_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'backimg_2.started')
                # update status
                backimg_2.status = STARTED
                backimg_2.setAutoDraw(True)
            
            # if backimg_2 is active this frame...
            if backimg_2.status == STARTED:
                # update params
                pass
            
            # *i2_txt* updates
            
            # if i2_txt is starting this frame...
            if i2_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                i2_txt.frameNStart = frameN  # exact frame index
                i2_txt.tStart = t  # local t and not account for scr refresh
                i2_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(i2_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'i2_txt.started')
                # update status
                i2_txt.status = STARTED
                i2_txt.setAutoDraw(True)
            
            # if i2_txt is active this frame...
            if i2_txt.status == STARTED:
                # update params
                pass
            
            # *pgnum* updates
            
            # if pgnum is starting this frame...
            if pgnum.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                pgnum.frameNStart = frameN  # exact frame index
                pgnum.tStart = t  # local t and not account for scr refresh
                pgnum.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pgnum, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pgnum.started')
                # update status
                pgnum.status = STARTED
                pgnum.setAutoDraw(True)
            
            # if pgnum is active this frame...
            if pgnum.status == STARTED:
                # update params
                pass
            
            # *key_resp_2* updates
            waitOnFlip = False
            
            # if key_resp_2 is starting this frame...
            if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_2.frameNStart = frameN  # exact frame index
                key_resp_2.tStart = t  # local t and not account for scr refresh
                key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_2.started')
                # update status
                key_resp_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_2.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_2.getKeys(keyList=['left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_2_allKeys.extend(theseKeys)
                if len(_key_resp_2_allKeys):
                    key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                    key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                    key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *backButton_2* updates
            
            # if backButton_2 is starting this frame...
            if backButton_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                backButton_2.frameNStart = frameN  # exact frame index
                backButton_2.tStart = t  # local t and not account for scr refresh
                backButton_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(backButton_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'backButton_2.started')
                # update status
                backButton_2.status = STARTED
                backButton_2.setAutoDraw(True)
            
            # if backButton_2 is active this frame...
            if backButton_2.status == STARTED:
                # update params
                pass
            
            # *nextButton_2* updates
            
            # if nextButton_2 is starting this frame...
            if nextButton_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                nextButton_2.frameNStart = frameN  # exact frame index
                nextButton_2.tStart = t  # local t and not account for scr refresh
                nextButton_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(nextButton_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'nextButton_2.started')
                # update status
                nextButton_2.status = STARTED
                nextButton_2.setAutoDraw(True)
            
            # if nextButton_2 is active this frame...
            if nextButton_2.status == STARTED:
                # update params
                pass
            
            # *start_2* updates
            
            # if start_2 is starting this frame...
            if start_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                start_2.frameNStart = frameN  # exact frame index
                start_2.tStart = t  # local t and not account for scr refresh
                start_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(start_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'start_2.started')
                # update status
                start_2.status = STARTED
                start_2.setAutoDraw(True)
            
            # if start_2 is active this frame...
            if start_2.status == STARTED:
                # update params
                pass
            # *mouse_3* updates
            
            # if mouse_3 is starting this frame...
            if mouse_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_3.frameNStart = frameN  # exact frame index
                mouse_3.tStart = t  # local t and not account for scr refresh
                mouse_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse_3.started', t)
                # update status
                mouse_3.status = STARTED
                mouse_3.mouseClock.reset()
                prevButtonState = mouse_3.getPressed()  # if button is down already this ISN'T a new click
            if mouse_3.status == STARTED:  # only update if started and not finished!
                buttons = mouse_3.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames([backButton_2,nextButton_2,start_2], namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(mouse_3):
                                gotValidClick = True
                                mouse_3.clicked_name.append(obj.name)
                        x, y = mouse_3.getPos()
                        mouse_3.x.append(x)
                        mouse_3.y.append(y)
                        buttons = mouse_3.getPressed()
                        mouse_3.leftButton.append(buttons[0])
                        mouse_3.midButton.append(buttons[1])
                        mouse_3.rightButton.append(buttons[2])
                        mouse_3.time.append(mouse_3.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Instruction2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instruction2" ---
        for thisComponent in Instruction2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Instruction2.stopped', globalClock.getTime(format='float'))
        # Run 'End Routine' code from code_5
        if 'backButton_2' in mouse_3.clicked_name or key_resp_2.keys == 'left':
            slideN -= 1
        elif 'nextButton_2' in mouse_3.clicked_name or key_resp_2.keys == 'right':
            slideN += 1
        elif 'start_2' in mouse_3.clicked_name or key_resp_2.keys == 'space':
            Instruct_Loop.finished = True
        
        # Constrain slideN within the valid range
        if slideN < minSlideN:
            slideN = minSlideN
        elif slideN > maxSlideN:
            slideN = maxSlideN
        
        if slideN == 1:
            instruct_txt = '在這個實驗中，\n\n你需要嘗試記住螢幕上顯示的數字。\n\n所有數字都在0到9之間。\n\n你會看到一串數字，依序顯示\n\n請記住整串數字'
        elif slideN == 2:
            instruct_txt = '一旦你記住這些數字後，你需要重述它們。\n\n在方塊中輸入剛剛顯示過的數字\n\n (空白鍵清除輸入的內容) \n\n 完成後請按 送出答案'
        # check responses
        if key_resp_2.keys in ['', [], None]:  # No response was made
            key_resp_2.keys = None
        Instruct_Loop.addData('key_resp_2.keys',key_resp_2.keys)
        if key_resp_2.keys != None:  # we had a response
            Instruct_Loop.addData('key_resp_2.rt', key_resp_2.rt)
            Instruct_Loop.addData('key_resp_2.duration', key_resp_2.duration)
        # store data for Instruct_Loop (TrialHandler)
        Instruct_Loop.addData('mouse_3.x', mouse_3.x)
        Instruct_Loop.addData('mouse_3.y', mouse_3.y)
        Instruct_Loop.addData('mouse_3.leftButton', mouse_3.leftButton)
        Instruct_Loop.addData('mouse_3.midButton', mouse_3.midButton)
        Instruct_Loop.addData('mouse_3.rightButton', mouse_3.rightButton)
        Instruct_Loop.addData('mouse_3.time', mouse_3.time)
        Instruct_Loop.addData('mouse_3.clicked_name', mouse_3.clicked_name)
        # the Routine "Instruction2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 500.0 repeats of 'Instruct_Loop'
    
    
    # set up handler to look after randomisation of conditions etc
    blocks = data.TrialHandler(nReps=4.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='blocks')
    thisExp.addLoop(blocks)  # add the loop to the experiment
    thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    
    for thisBlock in blocks:
        currentLoop = blocks
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # --- Prepare to start Routine "reset_correct" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('reset_correct.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from code_4
        incorrect_response = 0
        digitSpan += 1
        level = digitSpan
        thisExp.addData('level', level)
        # keep track of which components have finished
        reset_correctComponents = []
        for thisComponent in reset_correctComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "reset_correct" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in reset_correctComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "reset_correct" ---
        for thisComponent in reset_correctComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('reset_correct.stopped', globalClock.getTime(format='float'))
        # the Routine "reset_correct" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler(nReps=3.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='trials')
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        for thisTrial in trials:
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "set_item" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('set_item.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from code
            lower_bound = 10**(digitSpan - 1)
            upper_bound = 10**digitSpan - 1
            digits=randint(lower_bound,upper_bound)
            # keep track of which components have finished
            set_itemComponents = []
            for thisComponent in set_itemComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "set_item" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in set_itemComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "set_item" ---
            for thisComponent in set_itemComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('set_item.stopped', globalClock.getTime(format='float'))
            # the Routine "set_item" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            digitLoop = data.TrialHandler(nReps=digitSpan, method='sequential', 
                extraInfo=expInfo, originPath=-1,
                trialList=[None],
                seed=None, name='digitLoop')
            thisExp.addLoop(digitLoop)  # add the loop to the experiment
            thisDigitLoop = digitLoop.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisDigitLoop.rgb)
            if thisDigitLoop != None:
                for paramName in thisDigitLoop:
                    globals()[paramName] = thisDigitLoop[paramName]
            
            for thisDigitLoop in digitLoop:
                currentLoop = digitLoop
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisDigitLoop.rgb)
                if thisDigitLoop != None:
                    for paramName in thisDigitLoop:
                        globals()[paramName] = thisDigitLoop[paramName]
                
                # --- Prepare to start Routine "Digit_Presentation" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('Digit_Presentation.started', globalClock.getTime(format='float'))
                pres_text.setText(str(digits)[digitLoop.thisN])
                # keep track of which components have finished
                Digit_PresentationComponents = [backimg_3, image, Fixation, pres_text]
                for thisComponent in Digit_PresentationComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "Digit_Presentation" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 2.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *backimg_3* updates
                    
                    # if backimg_3 is starting this frame...
                    if backimg_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        backimg_3.frameNStart = frameN  # exact frame index
                        backimg_3.tStart = t  # local t and not account for scr refresh
                        backimg_3.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(backimg_3, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'backimg_3.started')
                        # update status
                        backimg_3.status = STARTED
                        backimg_3.setAutoDraw(True)
                    
                    # if backimg_3 is active this frame...
                    if backimg_3.status == STARTED:
                        # update params
                        pass
                    
                    # if backimg_3 is stopping this frame...
                    if backimg_3.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > backimg_3.tStartRefresh + 2-frameTolerance:
                            # keep track of stop time/frame for later
                            backimg_3.tStop = t  # not accounting for scr refresh
                            backimg_3.tStopRefresh = tThisFlipGlobal  # on global time
                            backimg_3.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'backimg_3.stopped')
                            # update status
                            backimg_3.status = FINISHED
                            backimg_3.setAutoDraw(False)
                    
                    # *image* updates
                    
                    # if image is starting this frame...
                    if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        image.frameNStart = frameN  # exact frame index
                        image.tStart = t  # local t and not account for scr refresh
                        image.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image.started')
                        # update status
                        image.status = STARTED
                        image.setAutoDraw(True)
                    
                    # if image is active this frame...
                    if image.status == STARTED:
                        # update params
                        pass
                    
                    # if image is stopping this frame...
                    if image.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > image.tStartRefresh + 2-frameTolerance:
                            # keep track of stop time/frame for later
                            image.tStop = t  # not accounting for scr refresh
                            image.tStopRefresh = tThisFlipGlobal  # on global time
                            image.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'image.stopped')
                            # update status
                            image.status = FINISHED
                            image.setAutoDraw(False)
                    
                    # *Fixation* updates
                    
                    # if Fixation is starting this frame...
                    if Fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        Fixation.frameNStart = frameN  # exact frame index
                        Fixation.tStart = t  # local t and not account for scr refresh
                        Fixation.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Fixation, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Fixation.started')
                        # update status
                        Fixation.status = STARTED
                        Fixation.setAutoDraw(True)
                    
                    # if Fixation is active this frame...
                    if Fixation.status == STARTED:
                        # update params
                        pass
                    
                    # if Fixation is stopping this frame...
                    if Fixation.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > Fixation.tStartRefresh + 1.0-frameTolerance:
                            # keep track of stop time/frame for later
                            Fixation.tStop = t  # not accounting for scr refresh
                            Fixation.tStopRefresh = tThisFlipGlobal  # on global time
                            Fixation.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'Fixation.stopped')
                            # update status
                            Fixation.status = FINISHED
                            Fixation.setAutoDraw(False)
                    
                    # *pres_text* updates
                    
                    # if pres_text is starting this frame...
                    if pres_text.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                        # keep track of start time/frame for later
                        pres_text.frameNStart = frameN  # exact frame index
                        pres_text.tStart = t  # local t and not account for scr refresh
                        pres_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(pres_text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pres_text.started')
                        # update status
                        pres_text.status = STARTED
                        pres_text.setAutoDraw(True)
                    
                    # if pres_text is active this frame...
                    if pres_text.status == STARTED:
                        # update params
                        pass
                    
                    # if pres_text is stopping this frame...
                    if pres_text.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > pres_text.tStartRefresh + 1.0-frameTolerance:
                            # keep track of stop time/frame for later
                            pres_text.tStop = t  # not accounting for scr refresh
                            pres_text.tStopRefresh = tThisFlipGlobal  # on global time
                            pres_text.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'pres_text.stopped')
                            # update status
                            pres_text.status = FINISHED
                            pres_text.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in Digit_PresentationComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "Digit_Presentation" ---
                for thisComponent in Digit_PresentationComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('Digit_Presentation.stopped', globalClock.getTime(format='float'))
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-2.000000)
            # completed digitSpan repeats of 'digitLoop'
            
            
            # --- Prepare to start Routine "Recall" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('Recall.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from fb_code
            # Initialize variables
            entered_text = ''
            enter_pressed = False  # Flag to track if Enter was pressed
            # Reset the TextBox text
            textbox.setText('')
            # Clear the event buffer to avoid skipping the routine due to earlier key presses
            event.clearEvents()
            
            textbox.reset()
            # setup some python lists for storing info about the mouse
            mouse.x = []
            mouse.y = []
            mouse.leftButton = []
            mouse.midButton = []
            mouse.rightButton = []
            mouse.time = []
            mouse.clicked_name = []
            gotValidClick = False  # until a click is received
            key_resp.keys = []
            key_resp.rt = []
            _key_resp_allKeys = []
            # keep track of which components have finished
            RecallComponents = [backimg_4, recall_txt, textbox, continue_button, clear_button, mouse, key_resp]
            for thisComponent in RecallComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Recall" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from fb_code
                # Check for key presses
                keys = event.getKeys()
                if mouse.isPressedIn(clear_button):
                    entered_text = ''
                    textbox.setText(entered_text)
                elif mouse.isPressedIn(continue_button):
                    continueRoutine = False
                for key in keys:
                    if key in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        entered_text += key
                        textbox.setText(entered_text)
                    elif key == 'left':
                        entered_text = entered_text[:-1]
                        textbox.setText(entered_text)
                    elif key == 'return' and not enter_pressed:
                        # End the routine when Enter is pressed
                        continueRoutine = False
                        enter_pressed = True  # Set flag to True
                    elif key == 'space':
                        entered_text = ''
                        textbox.setText(entered_text)
                
                
                
                # *backimg_4* updates
                
                # if backimg_4 is starting this frame...
                if backimg_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    backimg_4.frameNStart = frameN  # exact frame index
                    backimg_4.tStart = t  # local t and not account for scr refresh
                    backimg_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(backimg_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'backimg_4.started')
                    # update status
                    backimg_4.status = STARTED
                    backimg_4.setAutoDraw(True)
                
                # if backimg_4 is active this frame...
                if backimg_4.status == STARTED:
                    # update params
                    pass
                
                # *recall_txt* updates
                
                # if recall_txt is starting this frame...
                if recall_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    recall_txt.frameNStart = frameN  # exact frame index
                    recall_txt.tStart = t  # local t and not account for scr refresh
                    recall_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(recall_txt, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_txt.started')
                    # update status
                    recall_txt.status = STARTED
                    recall_txt.setAutoDraw(True)
                
                # if recall_txt is active this frame...
                if recall_txt.status == STARTED:
                    # update params
                    pass
                
                # *textbox* updates
                
                # if textbox is starting this frame...
                if textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox.frameNStart = frameN  # exact frame index
                    textbox.tStart = t  # local t and not account for scr refresh
                    textbox.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox.started')
                    # update status
                    textbox.status = STARTED
                    textbox.setAutoDraw(True)
                
                # if textbox is active this frame...
                if textbox.status == STARTED:
                    # update params
                    pass
                
                # *continue_button* updates
                
                # if continue_button is starting this frame...
                if continue_button.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    continue_button.frameNStart = frameN  # exact frame index
                    continue_button.tStart = t  # local t and not account for scr refresh
                    continue_button.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(continue_button, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'continue_button.started')
                    # update status
                    continue_button.status = STARTED
                    continue_button.setAutoDraw(True)
                
                # if continue_button is active this frame...
                if continue_button.status == STARTED:
                    # update params
                    pass
                
                # *clear_button* updates
                
                # if clear_button is starting this frame...
                if clear_button.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    clear_button.frameNStart = frameN  # exact frame index
                    clear_button.tStart = t  # local t and not account for scr refresh
                    clear_button.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(clear_button, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'clear_button.started')
                    # update status
                    clear_button.status = STARTED
                    clear_button.setAutoDraw(True)
                
                # if clear_button is active this frame...
                if clear_button.status == STARTED:
                    # update params
                    pass
                # *mouse* updates
                
                # if mouse is starting this frame...
                if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    mouse.frameNStart = frameN  # exact frame index
                    mouse.tStart = t  # local t and not account for scr refresh
                    mouse.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('mouse.started', t)
                    # update status
                    mouse.status = STARTED
                    mouse.mouseClock.reset()
                    prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
                if mouse.status == STARTED:  # only update if started and not finished!
                    buttons = mouse.getPressed()
                    if buttons != prevButtonState:  # button state changed?
                        prevButtonState = buttons
                        if sum(buttons) > 0:  # state changed to a new click
                            # check if the mouse was inside our 'clickable' objects
                            gotValidClick = False
                            clickableList = environmenttools.getFromNames([continue_button,], namespace=locals())
                            for obj in clickableList:
                                # is this object clicked on?
                                if obj.contains(mouse):
                                    gotValidClick = True
                                    mouse.clicked_name.append(obj.name)
                            x, y = mouse.getPos()
                            mouse.x.append(x)
                            mouse.y.append(y)
                            buttons = mouse.getPressed()
                            mouse.leftButton.append(buttons[0])
                            mouse.midButton.append(buttons[1])
                            mouse.rightButton.append(buttons[2])
                            mouse.time.append(mouse.mouseClock.getTime())
                
                # *key_resp* updates
                waitOnFlip = False
                
                # if key_resp is starting this frame...
                if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp.frameNStart = frameN  # exact frame index
                    key_resp.tStart = t  # local t and not account for scr refresh
                    key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.started')
                    # update status
                    key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp.getKeys(keyList=['1','2','3','4','5','6','7','8','9','backspace','delete','return'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_allKeys.extend(theseKeys)
                    if len(_key_resp_allKeys):
                        key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                        key_resp.rt = _key_resp_allKeys[-1].rt
                        key_resp.duration = _key_resp_allKeys[-1].duration
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in RecallComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Recall" ---
            for thisComponent in RecallComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('Recall.stopped', globalClock.getTime(format='float'))
            # Run 'End Routine' code from fb_code
            if textbox.text == str(digits):
                correct = 1
                
                # go to next level
                trials.finished = True
                fbTxt = '答對了!'
            else:
                correct = 0
                fbTxt = '答錯了!'
                incorrect_response +=1
            thisExp.addData('correct', correct)
            
            if trials.thisN+1 == trials.nTotal and incorrect_response == trials.nTotal:
                # end level and blocks
                last_level = level
                trials.finished = True
                blocks.finished = True
            else:
                last_level = level
            thisExp.addData('Response', entered_text)
            # store data for trials (TrialHandler)
            trials.addData('mouse.x', mouse.x)
            trials.addData('mouse.y', mouse.y)
            trials.addData('mouse.leftButton', mouse.leftButton)
            trials.addData('mouse.midButton', mouse.midButton)
            trials.addData('mouse.rightButton', mouse.rightButton)
            trials.addData('mouse.time', mouse.time)
            trials.addData('mouse.clicked_name', mouse.clicked_name)
            # check responses
            if key_resp.keys in ['', [], None]:  # No response was made
                key_resp.keys = None
            trials.addData('key_resp.keys',key_resp.keys)
            if key_resp.keys != None:  # we had a response
                trials.addData('key_resp.rt', key_resp.rt)
                trials.addData('key_resp.duration', key_resp.duration)
            # the Routine "Recall" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "Feedback" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('Feedback.started', globalClock.getTime(format='float'))
            feedback_txt.setText(fbTxt)
            # keep track of which components have finished
            FeedbackComponents = [backimg_5, feedback_txt]
            for thisComponent in FeedbackComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Feedback" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *backimg_5* updates
                
                # if backimg_5 is starting this frame...
                if backimg_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    backimg_5.frameNStart = frameN  # exact frame index
                    backimg_5.tStart = t  # local t and not account for scr refresh
                    backimg_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(backimg_5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'backimg_5.started')
                    # update status
                    backimg_5.status = STARTED
                    backimg_5.setAutoDraw(True)
                
                # if backimg_5 is active this frame...
                if backimg_5.status == STARTED:
                    # update params
                    pass
                
                # if backimg_5 is stopping this frame...
                if backimg_5.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > backimg_5.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        backimg_5.tStop = t  # not accounting for scr refresh
                        backimg_5.tStopRefresh = tThisFlipGlobal  # on global time
                        backimg_5.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'backimg_5.stopped')
                        # update status
                        backimg_5.status = FINISHED
                        backimg_5.setAutoDraw(False)
                
                # *feedback_txt* updates
                
                # if feedback_txt is starting this frame...
                if feedback_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    feedback_txt.frameNStart = frameN  # exact frame index
                    feedback_txt.tStart = t  # local t and not account for scr refresh
                    feedback_txt.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(feedback_txt, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_txt.started')
                    # update status
                    feedback_txt.status = STARTED
                    feedback_txt.setAutoDraw(True)
                
                # if feedback_txt is active this frame...
                if feedback_txt.status == STARTED:
                    # update params
                    pass
                
                # if feedback_txt is stopping this frame...
                if feedback_txt.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > feedback_txt.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        feedback_txt.tStop = t  # not accounting for scr refresh
                        feedback_txt.tStopRefresh = tThisFlipGlobal  # on global time
                        feedback_txt.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'feedback_txt.stopped')
                        # update status
                        feedback_txt.status = FINISHED
                        feedback_txt.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in FeedbackComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Feedback" ---
            for thisComponent in FeedbackComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('Feedback.stopped', globalClock.getTime(format='float'))
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 3.0 repeats of 'trials'
        
    # completed 4.0 repeats of 'blocks'
    
    
    # --- Prepare to start Routine "End" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('End.started', globalClock.getTime(format='float'))
    Thank_you.setText('您能夠記住的數列共有 ' + str(last_level-1) + '個')
    # setup some python lists for storing info about the mouse_2
    mouse_2.x = []
    mouse_2.y = []
    mouse_2.leftButton = []
    mouse_2.midButton = []
    mouse_2.rightButton = []
    mouse_2.time = []
    mouse_2.clicked_name = []
    gotValidClick = False  # until a click is received
    endKey.keys = []
    endKey.rt = []
    _endKey_allKeys = []
    # keep track of which components have finished
    EndComponents = [backimg_6, Thank_you, endImg, mouse_2, endKey]
    for thisComponent in EndComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "End" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *backimg_6* updates
        
        # if backimg_6 is starting this frame...
        if backimg_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            backimg_6.frameNStart = frameN  # exact frame index
            backimg_6.tStart = t  # local t and not account for scr refresh
            backimg_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(backimg_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'backimg_6.started')
            # update status
            backimg_6.status = STARTED
            backimg_6.setAutoDraw(True)
        
        # if backimg_6 is active this frame...
        if backimg_6.status == STARTED:
            # update params
            pass
        
        # *Thank_you* updates
        
        # if Thank_you is starting this frame...
        if Thank_you.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Thank_you.frameNStart = frameN  # exact frame index
            Thank_you.tStart = t  # local t and not account for scr refresh
            Thank_you.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Thank_you, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Thank_you.started')
            # update status
            Thank_you.status = STARTED
            Thank_you.setAutoDraw(True)
        
        # if Thank_you is active this frame...
        if Thank_you.status == STARTED:
            # update params
            pass
        
        # *endImg* updates
        
        # if endImg is starting this frame...
        if endImg.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            endImg.frameNStart = frameN  # exact frame index
            endImg.tStart = t  # local t and not account for scr refresh
            endImg.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endImg, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'endImg.started')
            # update status
            endImg.status = STARTED
            endImg.setAutoDraw(True)
        
        # if endImg is active this frame...
        if endImg.status == STARTED:
            # update params
            pass
        # *mouse_2* updates
        
        # if mouse_2 is starting this frame...
        if mouse_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_2.frameNStart = frameN  # exact frame index
            mouse_2.tStart = t  # local t and not account for scr refresh
            mouse_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_2.started', t)
            # update status
            mouse_2.status = STARTED
            mouse_2.mouseClock.reset()
            prevButtonState = mouse_2.getPressed()  # if button is down already this ISN'T a new click
        if mouse_2.status == STARTED:  # only update if started and not finished!
            buttons = mouse_2.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(endImg, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_2):
                            gotValidClick = True
                            mouse_2.clicked_name.append(obj.name)
                    x, y = mouse_2.getPos()
                    mouse_2.x.append(x)
                    mouse_2.y.append(y)
                    buttons = mouse_2.getPressed()
                    mouse_2.leftButton.append(buttons[0])
                    mouse_2.midButton.append(buttons[1])
                    mouse_2.rightButton.append(buttons[2])
                    mouse_2.time.append(mouse_2.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # *endKey* updates
        waitOnFlip = False
        
        # if endKey is starting this frame...
        if endKey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            endKey.frameNStart = frameN  # exact frame index
            endKey.tStart = t  # local t and not account for scr refresh
            endKey.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endKey, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'endKey.started')
            # update status
            endKey.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(endKey.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(endKey.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if endKey.status == STARTED and not waitOnFlip:
            theseKeys = endKey.getKeys(keyList=['space','return','escape'], ignoreKeys=["escape"], waitRelease=False)
            _endKey_allKeys.extend(theseKeys)
            if len(_endKey_allKeys):
                endKey.keys = _endKey_allKeys[-1].name  # just the last key pressed
                endKey.rt = _endKey_allKeys[-1].rt
                endKey.duration = _endKey_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in EndComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "End" ---
    for thisComponent in EndComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('End.stopped', globalClock.getTime(format='float'))
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_2.x', mouse_2.x)
    thisExp.addData('mouse_2.y', mouse_2.y)
    thisExp.addData('mouse_2.leftButton', mouse_2.leftButton)
    thisExp.addData('mouse_2.midButton', mouse_2.midButton)
    thisExp.addData('mouse_2.rightButton', mouse_2.rightButton)
    thisExp.addData('mouse_2.time', mouse_2.time)
    thisExp.addData('mouse_2.clicked_name', mouse_2.clicked_name)
    # check responses
    if endKey.keys in ['', [], None]:  # No response was made
        endKey.keys = None
    thisExp.addData('endKey.keys',endKey.keys)
    if endKey.keys != None:  # we had a response
        thisExp.addData('endKey.rt', endKey.rt)
        thisExp.addData('endKey.duration', endKey.duration)
    thisExp.nextEntry()
    # the Routine "End" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
