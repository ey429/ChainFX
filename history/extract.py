if __name__ == '__main__':
	f = open('USDJPY.csv', 'r')
	array = []
	for line in f:
		values = line.split(',')
		v_open   = eval(values[2])
		v_high   = eval(values[3])
		v_low    = eval(values[4])
		v_close  = eval(values[5])
		v_amount = eval(values[6])
		array.append((v_open, v_close, v_high, v_low, v_amount))
	f.close()
	
	pos = 0
	summary = []
	
	P = 5
	
	while pos + P <= len(array):
		S = 0
		A = 0
		H = 0
		L = 999
		
		v_open  = array[pos][0]
		v_close = array[pos + P - 1][1]
		
		for j in range(P):
			h = array[pos + j][2]
			l = array[pos + j][3]
			average = (h + l) / 2
			
			S += average * array[pos + j][4]
			A += array[pos + j][4]
			H  = max(H, h)
			L  = min(L, l)
		
		summary.append((S / A, v_open, v_close, H, L, A))
		pos += P
	
	M = 100000
	
	for j in range(len(summary) / M):
		f = open('train' + str(j + 1) + '.txt', 'w')
		for k in range(M):
			pos = j * M + k
			f.write(str(summary[pos]) + '\n')
		f.close()
		
	f = open('test.txt', 'w')
	pos = (len(summary) / M) * M
	while pos < len(summary):
		f.write(str(summary[pos]) + '\n')
		pos += 1
	f.close()
