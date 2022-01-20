# pawpularity_final_submission
my final submission for the pawpularity competition on kaggle. I have quite mixed feelings about this competition. 
Broadly I think they can be summarised as:

1) I was very happy to end up with a bronze medal. This was my third competition and my best result by miles, so 
obviously I'm pleased with that.

2) my final submission wasn't just a code-copy effort like most of my submissions. I actually did modify and play 
around with the code in order to come up with something that was kind of my own. I was pleased with this. Its progress

3) my final submission wasn't completely my own. I borrow HEAVILY from the work of others. The final submission was 
a bit of a mashup of code and ideas I borrowed from others. I honestly don't know if this is good or bad in an 
objective sense. I see that even the grandmasters are borrowing/copying from each other, so i guess in a way its quite normal. 
I don't feel great about it though. I would like to have a final submission that is really mine (if that's possible)

4) my final result was a LOT better than expected it would be. I jumped up about 270 places once the private 
leaderboard came out. this was not expected, but in hindsight it probably should have. Such a large chunk of the total 
data was held back for the private leaderboard, it was probably inevitable that there would be a bit shake up.

5) I learned a lot: training locally & uploading saved models for inference; using "bleeding edge" models instead 
of the models included in the normal pytorch modelzoo; building data loaders of different varieties; using 
albumentations and fastai tfms to create synthetic data... there is quite a list of new things that I learned. 
But TBH that just made me much more aware of how much MORE i need to learn.

6) compute resources - i have never been more conscious of how important it is to have decent resources on my local 
machine. I guess its more than reasonable to rent a GPU from AWS or other cloud based vendor, but I felt it was much 
better having a big fast GPU, fast CPU and lots of RAM locally. Its interesting that this coincided with the 
purchase of a new machine with greatly upgraded power. I wonder if I could have tried as many ideas if I 
was still using my old rtx2070. I can't say for certain, but I suspect not. Certainly it was very nice not to have out 
of memory errors and enormously reduced training times with the new machine.

7) Its interesting that ex3 and ex7 are both "fastai" based approaches. I didn't set out to do that, and there were 
about 8 other approaches I tried that were not fastai based, but ex3 stood out early on as an excellent performer, and
ex7 ended up being the best. There is a lot less code in both of these approaches than there was in some of my other 
efforts, and yet these two performed better. Something to be learned here.

Onwards and upwards. Lots more to learn. Time to focus on NLP for the feedback competition.
