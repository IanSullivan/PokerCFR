# PokerCFR

## Abstractions

#### Lossless:
Card rank is kept normal
Suit is ignored whenever possible
'o' means off-suit(flushes are impossible with this suit), 's' means suit is needed (possible flush)
#### Buckets
Clustering teqniques are to group similar hands together
EHS looks at the hand stength of the hand, Potential aware looks ahead so similar flushes and straits are placed together 
| EHS buckets      | Potential-aware Buckets         |
| ------------- |:-------------:|
| [6s, 8s][3s, As, Qo, To]      |[6s, 8s][3s, As, Qo, To]|
| [6s, 8s][7s, As, Ko, Qo]   |[6s, 9s][3s, 5o, Jo, Qs]|  
| [6s, 8s][4s, Ao, Qs, To] |[5s, 9s][4s, 7s, Ko, Qo]|  
| [6s, 8s][2s, 3s, 4o, Qo]     |[3s, 9s][2o, 8s, Js, Ko]|
| [6s, 8s][2s, 3o, 5o, Qs]    |[6s, 8s][4s, Js, Ko, To]

## Resources
##### Intro to Counter factual regret minimization: </br>
https://justinsermeno.com/posts/cfr/ </br>
http://modelai.gettysburg.edu/2013/cfr/cfr.pdf </br>
https://medium.com/ai-in-plain-english/steps-to-building-a-poker-ai-part-1-outline-and-history-58fbedaf6ded</br>

##### Limit Poker Solved: </br>
https://poker.cs.ualberta.ca/publications/2015-ijcai-cfrplus.pdf </br>
https://poker.cs.ualberta.ca/publications/heads-up_limit_poker_is_solved.acm2017.pdf </br>

##### Libratus: </br>
https://www.cs.cmu.edu/~noamb/papers/17-IJCAI-Libratus.pdf </br>
https://www.cs.cmu.edu/~noamb/papers/17-arXiv-Subgame.pdf </br>
https://www.cs.cmu.edu/~noamb/papers/18-NIPS-Depth.pdf </br>

##### Abstractions: </br>
https://nebula.wsimg.com/197ee65d8124f2060c45478c8080da7c?AccessKeyId=4F0E80116E133E66881C&disposition=0&alloworigin=1 </br>
https://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf </br>
https://www.cs.cmu.edu/~noamb/papers/14-AAAI-Regret.pdf </br>

##### Subgame solving: </br>
https://poker.cs.ualberta.ca/publications/aaai2014-cfrd.pdf </br>
https://dl.acm.org/doi/10.5555/3015812.3015898 </br>
https://poker.cs.ualberta.ca/publications/NIPS09-graft.pdf </br>

##### Evaluating: </br>
https://poker.cs.ualberta.ca/publications/kan.msc.pdf </br>
http://poker.cs.ualberta.ca/publications/aaai17ws-burch-aivat.pdf </br>
https://poker.cs.ualberta.ca/publications/jdavidson.msc.pdf </br>

