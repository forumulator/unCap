from __future__ import division
from scipy import misc, ndimage
import numpy as np
import math
import os, time
import cPickle as pickle 

feat_dict = dict()
proc_list = []

PIXEL_WHITE = 255
PIXEL_BLACK = 0

H_PARTS = 3
V_PARTS = 3

char_maxlen = 61
char_minlen = 32
char_avglen = 40

def get_odd(flnum):
	cnum = math.ceil(flnum)
	return cnum if (cnum % 2) else cnum - 1

# Change func logic, Set if white occurs in range 
# of whites, then it is white.
def crop_char(img, WIN_SIZE = 9):

	print("window size" + str(WIN_SIZE))
	new_chars = 0
	img_i = img.shape[0]
	img_j = img.shape[1]

	def chars_on(j, s, clen = char_minlen):
		lim = 0
		if s:
			lim = img_j 
		return int(abs(lim - j) / clen)

	
	WHITE_ARRAY = (np.zeros(img_i) +  255)

	logic_arr = []
	range_st = 0

	# Indicates whole captcha and not just seq of letters
	if (img_j > 200):
		logic_arr = [0  for i in range(10)]
		range_st = 10

	j = 0

	# Categorize each column as black or white
	for j in range(range_st, img_j - range_st):
		bflag = 0
		if ((img[:, j] != WHITE_ARRAY).any()):
			bflag = 1

		logic_arr.append(bflag)

	if (img_j > 200):
		logic_arr.extend([0  for i in range(range_st)])

	print(logic_arr)

	# No more seraration possible
	if (logic_arr == [1 for ii in range(len(logic_arr))]):
		yield np.zeros(shape(img_i, img_j))
		return

	# Window averaging
	flr = int(WIN_SIZE / 2)
	for j in range(10, img_j - 10):
		window = logic_arr[j - flr :j + flr + 1]
		avg_bit = sum(window) / WIN_SIZE
		avg_bit = 1 if (avg_bit >= 0.5) else 0

		logic_arr[j] = avg_bit


	print(logic_arr)

	j = 0
	char_idx = 0

	total_chars = chars_on(img_j, 0, char_avglen)
	while char_idx < total_chars:
		print(char_idx)
		while (j < img_j) and not logic_arr[j]:
			j += 1

		char_st = j
		while (j < img_j) and (logic_arr[j] or (j - char_st < char_minlen) \
			or (chars_on(j, 0) + chars_on(j, 1)) < total_chars):
			j += 1
		char_end = j

		char_len = char_end - char_st
		char_img = img[:, char_st:char_end]

		if (char_len):
			if (char_len > char_maxlen):
				print("Size exceeded " + str(char_len))
				misc.imshow(char_img)
				char_idx += chars_on(char_len, 0)
				for subchar in crop_char(char_img, get_odd(WIN_SIZE / 2)):
					yield subchar

			else:
				char_idx += 1
				misc.imshow(char_img)

				print("yielding " + str(char_idx))
				yield char_img


		if j >= img_j:
			print("Img fin")
			break


	return


def get_attr(char_img):

	global H_PARTS, V_PARTS
	# print("Char attributes")
	img_i = char_img.shape[0]
	img_j = char_img.shape[1]

	# print(char_img.shape)
	

	perc_arr = []

	for h in range(H_PARTS):
		h_st = int(math.ceil(img_j / H_PARTS) * h)
		h_end = int(min(math.ceil(img_j/ H_PARTS) * (h + 1), img_j))
		for v in range(V_PARTS):
			v_st = int(math.ceil(img_i / V_PARTS) * v)
			v_end = int(min(math.ceil(img_i/ V_PARTS) * (v + 1), img_i))

			window = char_img[v_st:v_end, h_st:h_end]
			# print(window.shape)
			black = 0
			total = 0
			for ii in range(window.shape[0]):
				for jj in range(window.shape[1]):
					total += 1
					if (window[ii][jj] != 255):
						black += 1

			perc_arr.append(round(black / total, 3))

	# print(perc_arr)
	# print("\n\n\n")
	return (img_j, perc_arr)

def clean_img(raw_img):


	def to_white(mat, size):
		flag = True
		for ii in range(size):
			for jj in range(size):
				if (outerbox(ii, jj, size) and mat[ii][jj] != PIXEL_WHITE):
					flag = False
					break

			if not flag:
				break

		return flag




	def outerbox(i, j, size):
		if (i == 0) or (j == 0):
			return True
		if (i == size -1 ) or (j== size -1):
			return True

		return False


	w1 = raw_img
	WFILTER = w1 > 200
	
	w1[WFILTER] = PIXEL_WHITE

	# misc.imshow(w1)

	FLT_SIZE = 7
	img_i = w1.shape[0]
	img_j = w1.shape[1]

	for i in range(img_i - FLT_SIZE):
		for j in range(img_j - FLT_SIZE):
			tm = w1[i:(i+FLT_SIZE), j:j+(FLT_SIZE)]
			# print(tm.shape)
			if (to_white(tm, FLT_SIZE) == True):
				tm[:, :] = PIXEL_WHITE

	w1 = ndimage.median_filter(w1, 3)
	BFILTER = w1 <= 200
	w1[WFILTER] = PIXEL_WHITE
	w1[BFILTER] = PIXEL_BLACK

	return w1



def train():

	global feat_dict, proc_list

	TRAIN_LIM = 20
	TRAIN_DIR = "/home/pranx/Downloads/train/"

	RES_FILES = ["proc.dat", "learnt.dat"]
	PLIST_FILE = TRAIN_DIR + "proc.dat"
	LEARNT_FILE = TRAIN_DIR + "learnt.dat"

	if os.path.exists(LEARNT_FILE):
		with open(LEARNT_FILE, "rb") as pfile:
			feat_dict = pickle.load(pfile)

	if os.path.exists(PLIST_FILE):
		with open(PLIST_FILE, "rb") as pfile:
			proc_list = pickle.load(pfile)

	print(proc_list)
	#  time.sleep(5)

	train_char = 0
	count = 1
	for file in os.listdir(TRAIN_DIR):

		if (file in proc_list) or (file in RES_FILES):
			continue

		if (train_char >= TRAIN_LIM):
			break

		w1 = misc.imread(TRAIN_DIR + file, True)
		w1 = clean_img(w1)


		misc.imshow(w1)

		# print("Enter Captcha Value:")
		# capt_val = raw_input()

		capt_val = file.split(".")[0].upper()
		char_idx = 0

		for char_img in crop_char(w1):
			this_char = capt_val[char_idx]

			char_j, flist = get_attr(char_img)

			if not this_char in feat_dict:
				
				# misc.imshow(char_img)
				feat_dict[this_char] = (char_j, flist)
				# print(feat_dict[this_char])
				# os.system("pause")
				train_chars += 1

			else:
				oj, oflist = feat_dict[this_char]
				flist = [round((flist[ii] + oflist[ii])/2, 3) for ii in range(H_PARTS*V_PARTS)]
				char_j = int((char_j + oj)/2)

				feat_dict[this_char] = (char_j, flist)

			print(feat_dict[this_char])

			char_idx += 1
			if (char_idx == 5):
				break


		proc_list.append(file)
		print("Processed image " + file)
		count += 1


	print("Pickling learnt data to file")
	# Pickle the data
	with open(LEARNT_FILE, "wb") as pfile:
		pickle.dump(feat_dict, pfile)

	with open(PLIST_FILE, "wb") as pf1:
		pickle.dump(proc_list, pf1)


def test():
	global feat_dict

	TRAIN_DIR = "/home/pranx/Downloads/train/"
	TEST_DIR = "/home/pranx/Downloads/test/"
	LEARNT_FILE = TRAIN_DIR + "learnt.dat"

	if os.path.exists(LEARNT_FILE):
		with open(LEARNT_FILE, "rb") as pfile:
			feat_dict = pickle.load(pfile)

	else:
		raise Exception("FileError: learnt.dat not found")

	cap_ans = ''

	TEST_IMG = TEST_DIR + "t1.png"

	t1 = misc.imread(TEST_IMG, True)
	t1 = clean_img(t1)

	for char_img in crop_char(t1):
		char_j, den_list = get_attr(char_img)
		min_err = 10000
		idx = ''

		for timg in feat_dict:
			tj, tdlist = feat_dict[timg]

			if (abs(char_j - tj) > 5):
				continue 

			this_err = sum([(den_list[ii] - tdlist[ii])**2 for \
					ii in range(H_PARTS * V_PARTS)])

			if (this_err < min_err):
				min_err = this_err
				idx = timg

		cap_ans += idx

	print("Text in Captcha is: " + cap_ans)




if __name__ == "__main__":
	train()
	test()

