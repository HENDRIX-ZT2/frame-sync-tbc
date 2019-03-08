import argparse
import ffmpeg
import logging
import numpy as np
import os
import subprocess
import pycorrelate as pyc
# import matplotlib.pyplot as plt
# from scipy.ndimage.interpolation import shift
from ringbuffer import RingBuffer

parser = argparse.ArgumentParser(description='Streaming ffmpeg to tensorflow & numpy')
parser.add_argument('in_filename', help='Input filename')
parser.add_argument('out_filename', help='Output filename')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def number(n):
	if type(n) == type("str"):
		if "/" in n:
			n = n.split("/")
			return int(n[0]) / int(n[1])
		elif "." in n:
			return float(n)
	try:
		return int(n)
	except:
		return n
		
def get_video_size(filename):
	logger.info('Getting video size for {!r}'.format(filename))
	probe = ffmpeg.probe(filename)
	video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
	# print(video_info)
	width = number(video_info['width'])
	height = number(video_info['height'])
	# pix_fmt = number(video_info['pix_fmt'])
	sample_aspect_ratio = number(video_info['sample_aspect_ratio'])
	display_aspect_ratio = video_info['display_aspect_ratio'].replace("/",":")
	try:
		field_order = video_info['field_order']
	except:
		print("WARNING: could not find field order in video info")
		field_order = None
	#r_frame_rate is apparently deprecated
	rate = number(video_info['avg_frame_rate'])
	# duration = number(video_info['duration'])
	# codec_time_base = number(video_info['codec_time_base'])
	return width, height, rate, display_aspect_ratio, field_order


def start_ffmpeg_process1(in_filename, rate):
	logger.info('Starting ffmpeg process1')
	args = (
		ffmpeg
		.input(in_filename)
		.output('pipe:', format='rawvideo', pix_fmt='rgb24', r=rate)
		.compile()
	)
	return subprocess.Popen(args, stdout=subprocess.PIPE)


def start_ffmpeg_process2(out_filename, width, height, rate, display_aspect_ratio):
	logger.info('Starting ffmpeg process2')
	args = (
		ffmpeg
		.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=rate)
		.output(out_filename, pix_fmt='yuv420p', r=rate, aspect=display_aspect_ratio, preset="ultrafast", crf=0)
		.overwrite_output()
		.compile()
	)
	return subprocess.Popen(args, stdin=subprocess.PIPE)


def read_frame(process1, width, height):
	logger.debug('Reading frame')

	# Note: RGB24 == 3 bytes per pixel.
	frame_size = width * height * 3
	in_bytes = process1.stdout.read(frame_size)
	if len(in_bytes) == 0:
		frame = None
	else:
		assert len(in_bytes) == frame_size
		frame = (
			np
			.frombuffer(in_bytes, np.uint8)
			.reshape([height, width, 3])
		)
	return frame

	
def write_frame(process2, frame):
	logger.debug('Writing frame')
	process2.stdin.write(
		frame.tobytes()
	)

def fields(frame):
	"""
	takes an RGB24 unit8 array,
	returns two RRGB24 uint8 arrays
	"""	
	return frame[0::2,:,:], frame[1::2,:,:]

def frame(a, b):
	"""
	takes an RGB24 unit8 array,
	returns two RRGB24 uint8 arrays
	"""	
	height, width, depth = a.shape
	frame = np.empty( (height*2, width, depth), dtype=a.dtype )
	frame[0::2,:,:] = a
	frame[1::2,:,:] = b
	return frame
	
def cross_corr(a, b):
	#max up/down shift
	N = 5
	#returns peak value and offset for the cross-correlation of a and b
	# res = np.correlate( a, b, mode="same")
	# offset = np.argmax(res) - len(res)//2
	
	#this is working and equivalent, but not significantly faster than the full numpy cross-correlation
	res = np.empty(N*2+1)
	#can't do negative lag so just shift them and piece both together
	first = pyc.ucorrelate(a, b, maxlag=N+1)
	second = pyc.ucorrelate(b, a, maxlag=N+1)
	res[:N+1]  = first[::-1]
	res[N+1:]  = second[1:]
	offset = np.argmax(res)-N
	# this is not equivalent
	# for i, x in enumerate(range(-N, N+1)):
		# # out[i] = np.sum(np.abs(a- np.roll(b, x) ))
		# out[i] = np.correlate( a, np.roll(b, x), mode="valid")
	return offset
	# return np.max(res), offset


def find_x_offset(field):
	MAX_INTENSITY = 30
	mean = np.median(field[20:-20,0:30], axis=(0, 2))
	res = np.abs( np.diff( np.clip(mean, 0, MAX_INTENSITY) ) )
	peak = np.argmax(res) - 13
	val = np.max(res)
	if abs(peak) > 2:
		peak = 0
	if val > 0.4*MAX_INTENSITY:
		return peak, val
	else:
		return 0, val

def find_top_offset(field):
	MAX_INTENSITY = 30
	mean = np.mean(field[0:15,20:-20], axis=(1, 2))
	res = np.diff( np.clip(mean, 0, MAX_INTENSITY) )
	peak = np.argmax(res) - 3
	val = np.max(res)
	if abs(peak) > 2:
		peak = 0
	if val > 0.4*MAX_INTENSITY:
		return peak, val
	else:
		return 0, val
	
def find_bottom_offset(field):
	MAX_INTENSITY = 30
	#reverse
	mean = np.mean(field[:-15:-1,20:-20], axis=(1, 2))
	res = np.diff( np.clip(mean, 0, MAX_INTENSITY) )
	#reverse again
	peak = -np.argmax(res) + 3
	val = np.max(res)
	if abs(peak) > 2:
		peak = 0
	if val > 0.4*MAX_INTENSITY:
		return peak, val
	else:
		return 0, val

def y_average(frame, x_clip=15, y_clip=0):
	return np.mean(frame[:,x_clip:-x_clip], axis=(1,2))
	
def x_average(frame, x_clip=0, y_clip=15):
	return np.mean(frame[y_clip:-y_clip,:], axis=(0,2))
	
def run(in_filename, out_filename):
	print(in_filename, out_filename)
	width, height, rate, display_aspect_ratio, field_order = get_video_size(in_filename)
	print(width, height, rate, display_aspect_ratio, field_order)
	process1 = start_ffmpeg_process1(in_filename, rate)
	process2 = start_ffmpeg_process2(out_filename, width, height, rate, display_aspect_ratio)
	i = 0
	N = 5
	y_shifts_a = []
	y_shifts_b = []
	x_shifts_a = []
	x_shifts_b = []
	shape = (height, width, 3)
	
	frame_buffer = RingBuffer(N, shape, np.uint8)
	while True:
		in_frame = read_frame(process1, width, height)
		if in_frame is None:
			logger.info('End of input stream')
			break
		frame_buffer.append(in_frame)
		logger.debug('Processing frame')
		i+=1
		#then the read is filled
		if i > N-1:
			# get mean frame
			frame_mean = np.mean(frame_buffer, axis = 0)
			frame_central = frame_buffer.get()
			frame_central_a, frame_central_b = fields(frame_central)
			frame_mean_a, frame_mean_b = fields(frame_mean)
			
			y_shift_a = cross_corr(y_average(frame_mean_a), y_average(frame_central_a))
			y_shift_b = cross_corr(y_average(frame_mean_b), y_average(frame_central_b))
			x_shift_a = cross_corr(x_average(frame_mean_a), x_average(frame_central_a))
			x_shift_b = cross_corr(x_average(frame_mean_b), x_average(frame_central_b))
			# y_shifts_a.append( y_shift_a )
			# y_shifts_b.append( y_shift_b )
			# x_shifts_a.append( x_shift_a )
			# x_shifts_b.append( x_shift_b )
		
			out_a = np.roll(frame_central_a, (y_shift_a, x_shift_a), axis=(0,1))
			out_b = np.roll(frame_central_b, (y_shift_b, x_shift_b), axis=(0,1))
			# out_a = shift(frame_central_a, (y_shift_a, x_shift_a, 0), mode="nearest", order=0)
			# out_b = shift(frame_central_b, (y_shift_b, x_shift_b, 0), mode="nearest", order=0)
			
			# write output
			out_frame = frame(out_a , out_b)
			write_frame(process2, out_frame)
		# if i == 5500:
			# break
	# flush end of buffer in the end
	for out_frame in frame_buffer.flush_frames():
		write_frame(process2, out_frame)
		
	# plt.plot(y_shifts_a)
	# plt.plot(y_shifts_b)
	# plt.plot(x_shifts_a)
	# plt.plot(x_shifts_b)
	# plt.show()
	
	print(frame_buffer)
	logger.info('Waiting for ffmpeg process1')
	process1.wait()

	logger.info('Waiting for ffmpeg process2')
	process2.stdin.close()
	process2.wait()


if __name__ == '__main__':
	args = parser.parse_args()
	run(args.in_filename, args.out_filename)
