// AI Song Generator using TensorFlow.js and Magenta.js

// Global variables
let musicRNN;
let musicVAE;
let player;
let currentAudioBuffer = null;

// Initialize models and player when the page loads
document.addEventListener('DOMContentLoaded', async () => {
    try {
        // Show loading status
        document.body.classList.add('loading');
        
        // Initialize the MusicRNN model for melody continuation
        musicRNN = new mm.MusicRNN('https://storage.googleapis.com/magentadata/js/checkpoints/music_rnn/melody_rnn');
        await musicRNN.initialize();
        
        // Initialize the MusicVAE model for melody generation
        musicVAE = new mm.MusicVAE('https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/mel_4bar_small_q2');
        await musicVAE.initialize();
        
        // Initialize the player
        player = new mm.Player();
        
        // Hide loading and show the interface
        document.body.classList.remove('loading');
        console.log('Models loaded successfully');
        
        // Add event listeners
        document.getElementById('generate-btn').addEventListener('click', generateSong);
        document.getElementById('download-btn').addEventListener('click', downloadSong);
    } catch (error) {
        console.error('Error initializing models:', error);
        alert('Failed to initialize AI models. Please try reloading the page.');
    }
});

// Function to convert lyrics to musical features
function lyricsToFeatures(lyrics, artistStyle) {
    // Create a seed based on the lyrics
    const seed = new mm.NoteSequence.Note({
        pitch: stringToSeed(lyrics) % 12 + 60, // Convert text to a pitch between C4 and B4
        startTime: 0,
        endTime: 0.5,
        velocity: 80
    });
    
    // Create a basic sequence with the seed note
    const seedSequence = {
        notes: [seed],
        totalTime: 0.5,
        tempos: [{qpm: getTempoForStyle(artistStyle)}],
        quantizationInfo: {stepsPerQuarter: 4}
    };
    
    return seedSequence;
}

// Helper function to convert string to a number (simple hash)
function stringToSeed(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) - hash) + str.charCodeAt(i);
        hash |= 0; // Convert to 32bit integer
    }
    return Math.abs(hash);
}

// Get appropriate tempo based on artist style
function getTempoForStyle(style) {
    switch (style) {
        case 'hiphop': return 85 + Math.random() * 15;
        case 'rock': return 110 + Math.random() * 30;
        case 'pop': return 100 + Math.random() * 20;
        case 'electronic': return 125 + Math.random() * 30;
        default: return 100;
    }
}

// Generate a drum pattern based on artist style
function generateDrumPattern(style, duration) {
    // This is a simplified version. In a real app, you'd use a pre-trained drum model
    const sequence = {
        notes: [],
        totalTime: duration,
        tempos: [{qpm: getTempoForStyle(style)}],
        quantizationInfo: {stepsPerQuarter: 4}
    };
    
    const numBars = Math.ceil(duration / 2);
    
    // Different drum patterns based on style
    let pattern;
    switch (style) {
        case 'hiphop':
            // Basic hip hop pattern (kick on 1 and 3, snare on 2 and 4)
            pattern = [36, 0, 38, 0, 36, 0, 38, 0]; // MIDI notes: 36=kick, 38=snare
            break;
        case 'rock':
            pattern = [36, 42, 38, 42, 36, 42, 38, 42]; // With hi-hat (42)
            break;
        case 'pop':
            pattern = [36, 42, 38, 42, 36, 0, 38, 42];
            break;
        case 'electronic':
            pattern = [36, 0, 38, 0, 36, 36, 38, 42];
            break;
        default:
            pattern = [36, 0, 38, 0, 36, 0, 38, 0];
    }
    
    // Generate notes for each bar
    for (let bar = 0; bar < numBars; bar++) {
        for (let i = 0; i < pattern.length; i++) {
            if (pattern[i] > 0) {
                sequence.notes.push({
                    pitch: pattern[i],
                    startTime: bar * 2 + i * 0.25,
                    endTime: bar * 2 + i * 0.25 + 0.25,
                    isDrum: true,
                    velocity: i % 2 === 0 ? 100 : 80
                });
            }
        }
    }
    
    return sequence;
}

// Function to generate the song
async function generateSong() {
    try {
        // Get user inputs
        const lyricsInput = document.getElementById('lyrics-input').value.trim();
        const artistStyle = document.getElementById('artist-style').value;
        
        // Validate input
        if (lyricsInput.length < 3) {
            alert('Please enter at least a few words for lyrics');
            return;
        }
        
        // Show loading
        document.getElementById('loading').classList.remove('hidden');
        document.getElementById('player-container').classList.add('hidden');
        
        // 1. Convert lyrics to musical features
        const seedSequence = lyricsToFeatures(lyricsInput, artistStyle);
        
        // 2. Generate a melody using MusicVAE
        const melodies = await musicVAE.sample(1);
        
        // 3. Continue the melody using MusicRNN
        const rnnSteps = 32;
        const temperature = 1.0;
        let generatedMelody = await musicRNN.continueSequence(melodies[0], rnnSteps, temperature);
        
        // 4. Generate drum pattern
        const drumPattern = generateDrumPattern(artistStyle, generatedMelody.totalTime);
        
        // 5. Combine melody and drums
        const combinedSequence = mm.sequences.concatenate([generatedMelody, drumPattern]);
        
        // 6. Add variations based on the lyrical content
        addVariationsFromLyrics(combinedSequence, lyricsInput);
        
        // 7. Convert to audio
        currentAudioBuffer = await mm.sequenceToAudioBuffer(combinedSequence);
        
        // 8. Update the player
        const audioPlayer = document.getElementById('audio-player');
        const audioUrl = URL.createObjectURL(bufferToWave(currentAudioBuffer));
        audioPlayer.src = audioUrl;
        
        // 9. Display visualization
        displayVisualization(combinedSequence);
        
        // 10. Show player and download button
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('player-container').classList.remove('hidden');
        document.getElementById('download-btn').classList.remove('hidden');
        
    } catch (error) {
        console.error('Error generating song:', error);
        document.getElementById('loading').classList.add('hidden');
        alert('Failed to generate song. Please try again with different inputs.');
    }
}

// Add variations to the sequence based on lyrics
function addVariationsFromLyrics(sequence, lyrics) {
    // This is a simplified version. In a real app, you'd use NLP to analyze lyrics
    // and make more sophisticated modifications
    
    // Extract some basic features from the lyrics
    const wordCount = lyrics.split(/\s+/).length;
    const averageWordLength = lyrics.replace(/\s+/g, '').length / wordCount;
    const hasExclamation = lyrics.includes('!');
    const isQuestion = lyrics.includes('?');
    
    // Adjust note velocities based on word count (more words = more dynamic)
    const velocityMod = Math.min(1.3, Math.max(0.8, wordCount / 10));
    
    // Adjust note lengths based on average word length
    const noteLengthMod = Math.min(1.5, Math.max(0.7, averageWordLength / 5));
    
    // Apply modifications to the sequence
    sequence.notes.forEach(note => {
        if (!note.isDrum) {
            // Modify velocity
            note.velocity = Math.min(127, Math.max(30, note.velocity * velocityMod));
            
            // Modify note length
            const duration = note.endTime - note.startTime;
            note.endTime = note.startTime + (duration * noteLengthMod);
            
            // Add special effects for questions or exclamations
            if (isQuestion && Math.random() > 0.7) {
                note.pitch = Math.min(84, note.pitch + 2); // Higher notes for questions
            }
            
            if (hasExclamation && Math.random() > 0.7) {
                note.velocity = Math.min(127, note.velocity + 20); // Louder for exclamations
            }
        }
    });
}

// Display visualization of the generated song
function displayVisualization(sequence) {
    const container = document.getElementById('visualization-container');
    container.innerHTML = '';
    
    // Use tfvis to create a visualization if available
    if (window.tfvis) {
        const surface = tfvis.visor().surface({ name: 'Note Sequence', tab: 'Song Visualization' });
        
        // Create a piano roll visualization
        mm.logging.resetAll();
        const div = document.createElement('div');
        container.appendChild(div);
        
        const visualizer = new mm.PianoRollCanvasVisualizer(sequence, div);
    } else {
        // Fallback to a simple visualization
        const canvas = document.createElement('canvas');
        canvas.width = 500;
        canvas.height = 200;
        container.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#f0f0f0';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw each note
        sequence.notes.forEach(note => {
            const x = (note.startTime / sequence.totalTime) * canvas.width;
            const width = ((note.endTime - note.startTime) / sequence.totalTime) * canvas.width;
            const y = canvas.height - ((note.pitch - 40) * 2);
            const height = note.isDrum ? 5 : 3;
            
            ctx.fillStyle = note.isDrum ? '#ff5733' : '#3366ff';
            ctx.fillRect(x, y, width, height);
        });
    }
}

// Convert AudioBuffer to WAV format for download
function bufferToWave(abuffer) {
    const numOfChan = abuffer.numberOfChannels;
    const length = abuffer.length * numOfChan * 2;
    const buffer = new ArrayBuffer(44 + length);
    const view = new DataView(buffer);
    const channels = [];
    let i, sample, offset = 0;

    // Write WAV header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + length, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numOfChan, true);
    view.setUint32(24, abuffer.sampleRate, true);
    view.setUint32(28, abuffer.sampleRate * numOfChan * 2, true);
    view.setUint16(32, numOfChan * 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, length, true);
    
    // Write interleaved data
    for (i = 0; i < abuffer.numberOfChannels; i++)
        channels.push(abuffer.getChannelData(i));
    
    while (offset < length) {
        for (i = 0; i < numOfChan; i++) {
            sample = Math.max(-1, Math.min(1, channels[i][offset / (numOfChan * 2)]));
            sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0; // convert to 16 bit
            view.setInt16(44 + offset, sample, true);
            offset += 2;
        }
    }
    
    return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

// Function to download the generated song
function downloadSong() {
    if (!currentAudioBuffer) {
        alert('No song has been generated yet');
        return;
    }
    
    const wavBlob = bufferToWave(currentAudioBuffer);
    const lyricsInput = document.getElementById('lyrics-input').value.trim();
    const artistStyle = document.getElementById('artist-style').value;
    
    // Create a file name based on the lyrics (first few words)
    const fileName = lyricsInput.substring(0, 20).replace(/\W+/g, '_') + '_' + artistStyle + '.wav';
    
    // Create download link
    const downloadLink = document.createElement('a');
    downloadLink.href = URL.createObjectURL(wavBlob);
    downloadLink.download = fileName;
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
}