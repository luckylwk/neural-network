import sys


def fn_print_epoch_progress( epoch, pct, cost ):
	pct *= 0.50
	sys.stdout.write('\r')
	sys.stdout.write('\t\tEpoch %d: [%-50s] %7.2f%% - Cost: %5.3f' % ( epoch, '*' * int(pct), 2.0*pct, cost ) )
	sys.stdout.flush()
	if pct==50.0: print '\r'
	##################