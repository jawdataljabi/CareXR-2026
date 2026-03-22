// AudioTranscriber.js
// Uses ASRModule on Spectacles to transcribe speech from wearer + nearby people.
// Prints whatever was heard in the last 1 second, then clears.

var asrModule;
try {
    asrModule = require('LensStudio:AsrModule');
} catch (e) {
    // ASRModule not available
}

script.createEvent('OnStartEvent').bind(function() {
    if (!asrModule) {
        print('[Transcriber] ASRModule not available on this platform.');
        return;
    }

    try {
        // Words confirmed final this cycle
        var confirmedWords = '';
        // Latest interim (unconfirmed) text from ASR
        var interimText = '';

        var options = AsrModule.AsrTranscriptionOptions.create();
        // 1.5s silence finalizes a segment, so phrases get committed quickly
        options.silenceUntilTerminationMs = 1500;
        // HighAccuracy gives best recognition for nearby/bystander speech
        options.mode = AsrModule.AsrMode.HighAccuracy;

        options.onTranscriptionUpdateEvent.add(function(eventArgs) {
            var text = eventArgs.text || '';
            if (eventArgs.isFinal) {
                // Segment is done: add to confirmed, clear interim
                if (text.trim().length > 0) {
                    confirmedWords += (confirmedWords.length > 0 ? ' ' : '') + text.trim();
                }
                interimText = '';
            } else {
                // Segment still in progress: update interim
                interimText = text;
            }
        });

        options.onTranscriptionErrorEvent.add(function(errorCode) {
            print('[Transcriber] ASR error: ' + errorCode);
        });

        asrModule.startTranscribing(options);
        print('[Transcriber] Listening...');

        var elapsed = 0;
        script.createEvent('UpdateEvent').bind(function(e) {
            elapsed += e.getDeltaTime();
            if (elapsed >= 1.0) {
                elapsed = 0;

                // Combine confirmed + any in-progress interim text
                var parts = [];
                if (confirmedWords.length > 0) parts.push(confirmedWords);
                if (interimText.trim().length > 0) parts.push(interimText.trim());

                var output = parts.join(' ').trim();
                if (output.length > 0) {
                    print('[Transcriber] ' + output);
                }

                // Full reset every second
                confirmedWords = '';
                interimText = '';
            }
        });
    } catch (e) {
        print('[Transcriber] ASR not available in preview: ' + e);
    }
});
