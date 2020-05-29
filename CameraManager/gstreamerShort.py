import contextlib
import sys
import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
from gi.repository import Gst, Gtk, GLib
Gst.init([])

@contextlib.contextmanager
def pull_sample(sink):
    sample = sink.emit('pull-sample')
    buf = sample.get_buffer()
    result, mapinfo = buf.map(Gst.MapFlags.READ)
    if result:
        yield sample, mapinfo.data
    buf.unmap(mapinfo)

def new_sample_callback(process,streamName):
    def callback(sink, pipeline):
        with pull_sample(sink) as (sample, data):
            #print(data)
            process(data, streamName)
        return Gst.FlowReturn.OK
    return callback

def on_bus_message(bus, message, pipeline, loop):
    if message.type == Gst.MessageType.EOS:
        seek_element = get_seek_element(pipeline)
        if loop and seek_element:
            flags = Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT
            if not seek_element.seek_simple(Gst.Format.TIME, flags, 0):
                Gtk.main_quit()
        else:
            Gtk.main_quit()
    elif message.type == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write('Warning: %s: %s\n' % (err, debug))
    elif message.type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write('Error: %s: %s\n' % (err, debug))
        Gtk.main_quit()

def on_sink_eos(sink, pipeline):
    overlay = pipeline.get_by_name('overlay')
    if overlay:
        overlay.set_eos()

def quit():
    Gtk.main_quit()

def run_pipeline(source,on_buffer,signals):
    # Create pipeline
    pipeline = source
    pipeline = Gst.parse_launch(pipeline)

    # Set up a pipeline bus watch to catch errors.
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message', on_bus_message, pipeline, False)

    for name, signals in signals.items():
        component = pipeline.get_by_name(name)
        if component:
            for signal_name, signal_handler in signals.items():
                component.connect(signal_name, signal_handler, pipeline)

    # Run pipeline.
    pipeline.set_state(Gst.State.PLAYING)
    try:
        Gtk.main()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)

    # Process all pending MainContext operations.
    while GLib.MainContext.default().iteration(False):
        pass
