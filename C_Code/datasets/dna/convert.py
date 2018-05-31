#!/usr/bin/env python

# convert things into svm-light format

#<line> .=. <target> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
#<target> .=. +1 | -1 | 0 | <float> 
#<feature> .=. <integer> | "qid"
#<value> .=. <float>
#<info> .=. <string> 

import optparse
import bz2
import sys
import os
import math

def read_label_line(flab):
	if flab:
		return flab.readline()
	else:
		return '0\n'

# binary features for dna
def convert_dna(outf, fdat, flab, num=-1):
	d=fdat.readline()
	l=read_label_line(flab)
	acgt=range(0,256)
	for i in xrange(256):
		acgt[i]=0
		if i==ord('C'):
			acgt[i]=1
		elif i==ord('G'):
			acgt[i]=2
		elif i==ord('T'):
			acgt[i]=3

	line=0
	while d and l and (num<0 or line<num):
		offs=1
		s=l[:-1]
		for i in xrange(len(d)-1):
			s+=" %d:1.0" % (offs+acgt[ord(d[i])])
			offs+=4
		outf.write(s + '\n')
		d=fdat.readline()
		l=read_label_line(flab)
		line+=1;

		if not line % 1000:
			sys.stderr.write( '\r%d' % line)

# 3-grams for webspam
def convert_webspam(outf, fdat, flab, num=-1):

	def ngramify(o, s):
		l=len(s)-3
		ostr=[ord(i) for i in s]

		lst=[ (((ostr[i]<<8)+ostr[i+1])<<8)+ostr[i+2] for i in xrange(l) ]
		lst.sort()

		ngrams=list()

		j=0
		sum=0
		for i in xrange(l):
			ngram=None
			if i==l-1:
				if lst[j]!=lst[i]:
					ngram=(lst[j], i-j)
				else:
					ngram=(lst[i], 1)
			else:
				if lst[j]!=lst[i+1]:
					ngram=(lst[j], i-j+1)
					j=i+1
			if ngram:
				sum+=ngram[1]*ngram[1]
				ngrams.append(ngram)

		sum=math.sqrt(float(sum))

		if sum==0.0: # trap divide by zero
			sum=1.0

		for ngram in ngrams:
			o+=" %d:%g" % (ngram[0]+1, ngram[1]/sum)
		outf.write(o + '\n')


	line=0
	d=''
	while (num<0 or line<num):
		r='x'
		while r and r.find('\0') == -1:
			r=fdat.read(1048576)
			if r:
				d+=r

		start=0
		while (num<0 or line<num):
			idx=d.find('\0',start)
			if idx==-1:
				d=d[(start+1):]
				break
			else:
				l=read_label_line(flab)
				o=l[:-1]
				ngramify(o, d[start:idx])
				start=idx+1
				line+=1;

		sys.stderr.write( '\rPROGRESS: %d' % line)

		#eof
		if not r:
			break


# convert image
def convert_img(outf, fdat, flab, dims, normalize=False, num=-1):
	d=fdat.read(dims)
	l=read_label_line(flab)

	line=0
	while d and l and (num<0 or line<num):
		s=l[:-1]
		ostr=[ord(i) for i in d]
		if normalize:
			sum=0.0
			for i in xrange(len(ostr)-1):
				sum+=ostr[i]*ostr[i]
			sum=math.sqrt(sum)

			if sum==0.0: # trap divide by zero
				sum=1.0

			for i in xrange(len(ostr)-1):
				s+=" %d:%g" % (i+1, ostr[i]/sum)
		else:
			for i in xrange(len(ostr)-1):
				s+=" %d:%d" % (i+1, ostr[i])
		outf.write(s + '\n')
		d=fdat.read(dims)
		l=read_label_line(flab)
		line+=1;

		if not line % 1000:
			sys.stderr.write( '\r%d' % line)

# convert ascii
def convert_dense_ascii(outf, fdat, flab, normalize=False, num=-1):
	d=fdat.readline()
	l=read_label_line(flab)

	line=0
	while d and l and (num<0 or line<num):
		s=l[:-1]
		ostr=d.strip().split(' ')
		if normalize:
			sum=0.0
			for i in xrange(len(ostr)):
				x=float(ostr[i])
				sum+=x*x
			sum=math.sqrt(sum)

			if sum==0.0: # trap divide by zero
				sum=1.0

			for i in xrange(len(ostr)):
				s+=" %s:%g" % (i+1, float(ostr[i])/sum)
		else:
			for i in xrange(len(ostr)):
				s+=" %s:%s" % (i+1, ostr[i])
		outf.write(s + '\n')
		d=fdat.readline()
		l=read_label_line(flab)
		line+=1;

		if not line % 1000:
			sys.stderr.write( '\r%d' % line)

def parse_options():
	parser = optparse.OptionParser(usage="%prog [options] {dna|webspam|ocr|fd|alpha|beta|gamma|delta|epsilon|zeta} {train|val|test}\n\n"
			"script convert things into svm-light format")

	parser.add_option("-o", "--outfile", type="string", default='-',
			help="""File to write the results to, default is stdout""")
	parser.add_option("-n", '--num', type="int", default=-1,
			help="extract only num many examples")
	(options, args) = parser.parse_args()

	if len(args) != 2:
		parser.error("incorrect number of arguments")

	fnames= (args[0]+'_'+args[1]+'.dat.bz2', args[0]+'_'+args[1]+'.lab.bz2')
	files=list()

	for f in fnames:
		if not os.path.isfile(f):
			if f.find('lab')>0:
				sys.stderr.write("error: no label file found, filling with zeros\n")
				files.append(None)
			else:
				parser.error("dataset %s does not exist.\n\n"
						"script must be run in the path where the data is located.\n\n" % f)
		else:
			files.append(bz2.BZ2File(f))
	if options.outfile == '-':
		outf=sys.stdout
	else:
		outf=file(options.outfile,'wb')

	return (outf, files[0], files[1], args[0], options.num)


if __name__ == '__main__':
	outf, fdat,flab,d,num=parse_options()

	if d=='dna':
		convert_dna(outf, fdat,flab, num)
	elif d=='webspam':
		convert_webspam(outf, fdat,flab, num)
	elif d=='ocr':
		convert_img(outf, fdat,flab, 1156, True, num)
	elif d=='fd':
		convert_img(outf, fdat,flab, 900, True, num)
	elif d in ('alpha', 'beta', 'epsilon', 'zeta'):
		convert_dense_ascii(outf, fdat,flab, True, num)
	elif d in ('gamma', 'delta'):
		convert_dense_ascii(outf, fdat,flab, False, num)
