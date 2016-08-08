import re
import numpy as np
import pickle
from dateutil.parser import parse

months = set([	'Jan', 'Feb', 'Mar', 'Apr',
				'May', 'Jun', 'Jul', 'Aug',
				'Sep', 'Oct', 'Nov', 'Dec'])
home_status = set(['RENT', 'OWN', 'MORTGAGE'])
income_status = set(['VERIFIED - income',
					'not verified',
					'VERIFIED - income source'])

with open('Data for Cleaning & Modeling.csv') as f:
	
	data = [] 
	headers = f.readline()
	for line in f:
		line = line.strip()

		# deals with loan amount e.g. "$25,000"
		while True:
			match = re.search('"\$(\d+)(,\d+)+"', line)
			if not match:
				break
			money = match.group()

			money_literal = money.replace('$', '')
			money_literal = money_literal.replace(',', '')
			money_literal = money_literal.replace('"', '')
			line = line.replace(money, money_literal)

		csv = line.split(',')
		tmp = list(csv[:9])
		tmp.extend([''])
	
		# extract X11 through X15 
		for i in range(len(csv[10:])):
			item = csv[10 + i]

			if item in home_status:
				tmp.extend(csv[10 + i - 1 : 10 + i + 4])
				break
			elif item in income_status:
				tmp.extend(csv[10 + i - 3 : 10 + i + 2])
				break
			elif re.match('^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\-\d\d$', item):
				tmp.extend(csv[10 + i - 4 : 10 + i + 1])
				break

		if len(tmp) == 10:
			tmp.extend(['', '', '', '', ''])

		# X16: ignored
		tmp.extend([''])
		# X17 through X32
		tmp.extend(csv[-16:])
		csv = tmp

		if len(csv[0]) == 0 or len(csv) != 32:
			continue

		if (csv[11].strip() != '' and csv[11] not in home_status) or (csv[13].strip() != '' and csv[13] not in income_status): 
			continue

		# X1: Target
		csv[0] = csv[0][:-1]

		# X4: Loan amount requested
		csv[3] = csv[3][1:] if len(csv[3]) > 0 and csv[3][0] == '$' else csv[3]

		# X5: Loan amount funded
		csv[4] = csv[4][1:] if len(csv[4]) > 0 and csv[4][0] == '$' else csv[4]

		# X6: Investor-funded portion of loan
		csv[5] = csv[5][1:] if len(csv[5]) > 0 and csv[5][0] == '$' else csv[5]

		# X7: Number of payments (36 or 60)
		csv[6] = '36' if csv[6].find('36') >= 0 else '60'

		# X11: Number of years employed (0 to 10; 10 = 10 or more)
		if re.search('<(\s)*1', csv[10]):
			csv[10] = '0'
		else:
			csv[10] = re.search('(\d+)', csv[10]).group() if re.search('(\d+)', csv[10]) else ''

		# X13: Annual income of borrower
		csv[12] = csv[12][1:] if len(csv[12]) > 0 and csv[12][0] == '$' else csv[12]

		# X15: Date loan was issued
		if len(csv[14]) > 0:

			(month, year) = csv[14].split('-')
			# year 20xx
			if year[0] == '0' or year[0] == '1':
				year = '20' + year
			# year 19xx
			else:
				year = '19' + year
			# convert to 1st day of that month
			csv[14] = '-'.join(['1', month, year])


		# X23: Date the borrower's earliest reported credit line was opened
		if len(csv[22]) > 0:

			(month, year) = csv[22].split('-')
			# year 20xx
			if year[0] == '0' or year[0] == '1':
				year = '20' + year
			# year 19xx
			else:
				year = '19' + year
			# convert to 1st day of that month
			csv[22] = '-'.join(['1', month, year])

		# X30: "Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit."
		csv[29] = csv[29][:-1]

		csv = map(lambda x : x.strip(), csv)
		data.append(csv)

	pickle.dump(data, open('data_train.pickle', 'w'))
