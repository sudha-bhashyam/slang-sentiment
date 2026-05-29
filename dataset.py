"""
SIT770 — Unified Evaluation Dataset
Temporal Semantic Drift in Sentiment Classification of Social Media Slang

500 labelled sentences spanning:
  - Valence Reversal    (words whose polarity has flipped, e.g. fire, nasty)
  - Community Drift     (neologisms absent pre-2020, e.g. bussin, goated)
  - Polysemy Amplification (words with new senses, e.g. mid, cooked)
  - Literal controls    (same words used in literal/non-slang sense)
  - Mixed / hard cases  (competing signals)

Labels: 0 = negative, 1 = neutral, 2 = positive
Drift types: "valence", "community", "polysemy", "literal", "control", "mixed"

Used for:
  Task 2.2D — empirical motivation (baseline model evaluation)
  Task 2.3HD — full evaluation set

References supporting taxonomy:
  Valence Reversal:
    Wijaya & Yeniterzi (2011) — Understanding Semantic Change of Words Over Centuries
    Hamilton et al. (2016)   — Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change
  Community Drift:
    Eisenstein et al. (2014) — Diffusion of Lexical Change in Social Media
    Nguyen et al. (2020)     — BERTweet: A Pre-trained Language Model for English Tweets
  Polysemy Amplification:
    Loureiro et al. (2022)   — TempoWiC: An Evaluation Benchmark for Detecting Meaning Shift
  Temporal drift overall:
    Loureiro et al. (2022)   — TimeLMs: Diachronic Language Models from Twitter
    Tahmasebi et al. (2021)  — Survey of Computational Approaches to Lexical Semantic Change
"""

# ══════════════════════════════════════════════════════════════════════
# CATEGORY 1: VALENCE REVERSAL — positive slang (label=2)
# Words: fire, sick, nasty, dead, unhinged, insane, crazy, wild,
#        savage, dirty, wicked, stupid, killed, slaps, banger, hard
# Linguistic grounding: Wijaya & Yeniterzi (2011), Hamilton et al. (2016)
# ══════════════════════════════════════════════════════════════════════
VALENCE_REVERSAL_POS = [
    # FIRE
    {"text": "That concert was absolutely fire, crowd was electric all night",       "label": 2, "drift": "valence"},
    {"text": "Her cooking is fire, best meal I have had in years",                   "label": 2, "drift": "valence"},
    {"text": "This track is fire, been playing it every single day",                 "label": 2, "drift": "valence"},
    {"text": "That speech was fire, had everyone on their feet",                     "label": 2, "drift": "valence"},
    {"text": "His game last night was fire, scored four times easily",               "label": 2, "drift": "valence"},
    {"text": "That new album is absolutely fire best thing I heard all year",        "label": 2, "drift": "valence"},
    {"text": "the concert was absolutely fire from start to end",                    "label": 2, "drift": "valence"},
    {"text": "beat is hard and the bars are fire cannot ask for more",               "label": 2, "drift": "valence"},
    {"text": "that verse is absolutely fire top five bars of the year easy",         "label": 2, "drift": "valence"},
    {"text": "bro that verse was fire hardest thing I heard all month",              "label": 2, "drift": "valence"},
    {"text": "absolutely fire set loved every second of that show",                  "label": 2, "drift": "valence"},
    {"text": "the hook is fire love how it slaps in the car",                        "label": 2, "drift": "valence"},
    {"text": "stupid talented musician every song is fire no debate",                "label": 2, "drift": "valence"},
    {"text": "new EP is fire as anything every track certified banger",              "label": 2, "drift": "valence"},
    {"text": "that opening track was fire set the tone for the whole album",         "label": 2, "drift": "valence"},
    # SICK
    {"text": "That trick was sick, cannot believe he landed it perfectly",           "label": 2, "drift": "valence"},
    {"text": "Her art is so sick, everyone is obsessed with her style",              "label": 2, "drift": "valence"},
    {"text": "That film was sick, best thing I have watched this year",              "label": 2, "drift": "valence"},
    {"text": "The crowd went sick for that drop, energy was unreal all night",       "label": 2, "drift": "valence"},
    {"text": "that guitar solo was absolutely sick bro",                             "label": 2, "drift": "valence"},
    {"text": "the choreography was sick every move was perfect",                     "label": 2, "drift": "valence"},
    {"text": "absolutely sick trick he nailed every single rotation",                "label": 2, "drift": "valence"},
    {"text": "sick performance every move was calculated and perfect",               "label": 2, "drift": "valence"},
    {"text": "sick trick nailed every rotation perfectly unreal skill",              "label": 2, "drift": "valence"},
    {"text": "that guitar riff is SICK been on repeat since it dropped",             "label": 2, "drift": "valence"},
    # NASTY
    {"text": "That dunk was nasty, crowd erupted immediately after",                 "label": 2, "drift": "valence"},
    {"text": "His footwork is nasty, best player on the field by far",               "label": 2, "drift": "valence"},
    {"text": "That solo was nasty, guitarist absolutely shredded it",                "label": 2, "drift": "valence"},
    {"text": "That drop was nasty crowd went absolutely wild",                       "label": 2, "drift": "valence"},
    {"text": "that dunk was nasty the whole crowd lost it completely",               "label": 2, "drift": "valence"},
    {"text": "that nasty dunk had the whole arena shaking fr",                       "label": 2, "drift": "valence"},
    {"text": "the drop was nasty the energy went through the roof",                  "label": 2, "drift": "valence"},
    {"text": "that drop was NASTY bro the whole pit went absolutely wild",           "label": 2, "drift": "valence"},
    {"text": "absolutely nasty dunk had the whole arena on their feet",              "label": 2, "drift": "valence"},
    {"text": "filthy assist from him striker had no idea where ball came from",      "label": 2, "drift": "valence"},
    # INSANE / CRAZY / WILD / UNHINGED
    {"text": "The production on this album is insane, every track delivers",         "label": 2, "drift": "valence"},
    {"text": "She went crazy on that verse, best bars of the year",                  "label": 2, "drift": "valence"},
    {"text": "The choreography was unhinged in the best possible way",               "label": 2, "drift": "valence"},
    {"text": "His jumping ability is insane, defies physics honestly",               "label": 2, "drift": "valence"},
    {"text": "That plot twist was crazy, did not see it coming at all",              "label": 2, "drift": "valence"},
    {"text": "The energy in that room was wild, everyone was losing it",             "label": 2, "drift": "valence"},
    {"text": "That comeback was wild, no one expected them to win fr",               "label": 2, "drift": "valence"},
    {"text": "that comeback was absolutely insane the crowd erupted",                "label": 2, "drift": "valence"},
    {"text": "insane visuals the production was wild from start to finish",          "label": 2, "drift": "valence"},
    {"text": "bro that drop was unhinged in the best way",                           "label": 2, "drift": "valence"},
    {"text": "she killed every note the crowd went crazy",                           "label": 2, "drift": "valence"},
    {"text": "that was an insane game the energy was wild throughout",               "label": 2, "drift": "valence"},
    {"text": "the vibe was unhinged everyone was losing their minds",                "label": 2, "drift": "valence"},
    {"text": "crazy talented whole band went stupid hard tonight",                   "label": 2, "drift": "valence"},
    {"text": "crowd went absolutely wild when the drop hit insane energy",           "label": 2, "drift": "valence"},
    # STUPID / SAVAGE / DIRTY / WICKED / HARD / KILLED / SLAPS / BANGER
    {"text": "That pass was stupid accurate, hit him perfectly in stride",           "label": 2, "drift": "valence"},
    {"text": "She is stupid talented, everything she touches is gold",               "label": 2, "drift": "valence"},
    {"text": "The bass on this song is stupid heavy, shakes the whole room",         "label": 2, "drift": "valence"},
    {"text": "that crossover was savage he dropped him clean",                       "label": 2, "drift": "valence"},
    {"text": "that tackle was savage he just read the whole play perfectly",         "label": 2, "drift": "valence"},
    {"text": "that trick was dirty the skaters went wild",                           "label": 2, "drift": "valence"},
    {"text": "that dunk was dirty the commentators lost it completely",              "label": 2, "drift": "valence"},
    {"text": "that wicked show had everyone screaming throughout",                   "label": 2, "drift": "valence"},
    {"text": "wicked choreography every move was sick and precise",                  "label": 2, "drift": "valence"},
    {"text": "this beat is hard, been on repeat all day",                            "label": 2, "drift": "valence"},
    {"text": "the bass on this track goes stupid hard",                              "label": 2, "drift": "valence"},
    {"text": "the beat is so hard it literally shook the venue",                     "label": 2, "drift": "valence"},
    {"text": "she killed the stage performance was insane",                          "label": 2, "drift": "valence"},
    {"text": "killed it from first note to last note no breaks at all",              "label": 2, "drift": "valence"},
    {"text": "she killed every note I was not ready for that level",                 "label": 2, "drift": "valence"},
    {"text": "the track slaps so hard can't stop listening to it",                   "label": 2, "drift": "valence"},
    {"text": "the whole album slaps no weak tracks anywhere at all",                 "label": 2, "drift": "valence"},
    {"text": "whole album slaps so hard not a single skip anywhere",                 "label": 2, "drift": "valence"},
    {"text": "new album is a banger from start to finish",                           "label": 2, "drift": "valence"},
    {"text": "banger after banger this album is insane quality",                     "label": 2, "drift": "valence"},
    {"text": "that banger of an opener set the tone for everything",                 "label": 2, "drift": "valence"},
    {"text": "that was a banger live set slapped even harder in person",             "label": 2, "drift": "valence"},
    {"text": "the show was wild from the opening act to the encore",                 "label": 2, "drift": "valence"},
    {"text": "absolutely wild set the best live show I have seen",                   "label": 2, "drift": "valence"},
    {"text": "the verse was insane every bar landed perfectly no skips",             "label": 2, "drift": "valence"},
    {"text": "filthy footwork the defender had no idea what happened",               "label": 2, "drift": "valence"},
    {"text": "dirty footwork defender had absolutely no idea what hit",              "label": 2, "drift": "valence"},
    {"text": "the hook goes so hard been on repeat three days straight",             "label": 2, "drift": "valence"},
    {"text": "insane energy from the crowd the whole night was wild",                "label": 2, "drift": "valence"},
    {"text": "savage comeback commentators completely lost it fr",                   "label": 2, "drift": "valence"},
    {"text": "wicked visuals whole production was genuinely next level",             "label": 2, "drift": "valence"},
    {"text": "stupid good performance everyone left completely shook",               "label": 2, "drift": "valence"},
]

# ══════════════════════════════════════════════════════════════════════
# CATEGORY 2: COMMUNITY DRIFT — positive slang (label=2)
# Words: bussin, goated, no cap, slay, ate, rizz, lowkey, highkey,
#        periodt, fr fr, main character, understood the assignment,
#        sending me, based, hit different
# Linguistic grounding: Eisenstein et al. (2014)
# ══════════════════════════════════════════════════════════════════════
COMMUNITY_DRIFT_POS = [
    {"text": "This food is bussin, best burger I have ever had",                     "label": 2, "drift": "community"},
    {"text": "That set was bussin, every song was a banger",                         "label": 2, "drift": "community"},
    {"text": "This new restaurant is bussin, queue around the block",                "label": 2, "drift": "community"},
    {"text": "that performance was bussin bussin absolutely incredible",              "label": 2, "drift": "community"},
    {"text": "bro that set was bussin from the first note",                          "label": 2, "drift": "community"},
    {"text": "fr fr this drop is bussin harder than expected",                       "label": 2, "drift": "community"},
    {"text": "sending me with this track honestly bussin fr",                        "label": 2, "drift": "community"},
    {"text": "just heard the new drop bestie it is absolutely bussin no cap",        "label": 2, "drift": "community"},
    {"text": "highkey bussin meal the chef really understood the assignment",        "label": 2, "drift": "community"},
    {"text": "Her fashion sense is goated, everyone copies her style",               "label": 2, "drift": "community"},
    {"text": "That player is goated, carries the whole team every game",             "label": 2, "drift": "community"},
    {"text": "this artist is goated no debate at all",                               "label": 2, "drift": "community"},
    {"text": "goated performance from start to finish",                              "label": 2, "drift": "community"},
    {"text": "goated comeback no cap that verse was insane",                         "label": 2, "drift": "community"},
    {"text": "no cap the mixing on this album is goated",                            "label": 2, "drift": "community"},
    {"text": "lowkey the goated album of 2023 not taking any arguments",             "label": 2, "drift": "community"},
    {"text": "No cap that was the best show I have been to all year",                "label": 2, "drift": "community"},
    {"text": "no cap this is the best album of the year",                            "label": 2, "drift": "community"},
    {"text": "no cap this concert was life changing honestly",                       "label": 2, "drift": "community"},
    {"text": "she really ate with that breakdown no cap",                            "label": 2, "drift": "community"},
    {"text": "periodt this is the best show I have ever attended",                   "label": 2, "drift": "community"},
    {"text": "the whole album ate periodt no skips anywhere",                        "label": 2, "drift": "community"},
    {"text": "she slayed every single look backstage periodt",                       "label": 2, "drift": "community"},
    {"text": "She slayed that performance, audience gave a standing ovation",        "label": 2, "drift": "community"},
    {"text": "slay queen that performance was everything",                           "label": 2, "drift": "community"},
    {"text": "she slay'd every single second of that performance fr fr",             "label": 2, "drift": "community"},
    {"text": "she ate and left no crumbs best performance of the year",              "label": 2, "drift": "community"},
    {"text": "she ate and left no crumbs wow",                                       "label": 2, "drift": "community"},
    {"text": "She ate that role, critics are raving about her performance",          "label": 2, "drift": "community"},
    {"text": "she really ate with that breakdown fr the crowd lost it",              "label": 2, "drift": "community"},
    {"text": "she understood the assignment, delivered exactly what was needed",     "label": 2, "drift": "community"},
    {"text": "she really understood the assignment tonight",                         "label": 2, "drift": "community"},
    {"text": "understood the assignment and then LEFT with it incredible",           "label": 2, "drift": "community"},
    {"text": "goated performance no cap understood the assignment from minute one",  "label": 2, "drift": "community"},
    {"text": "Lowkey obsessed with this song, cannot get it out of my head",         "label": 2, "drift": "community"},
    {"text": "lowkey this is the best decision I ever made",                         "label": 2, "drift": "community"},
    {"text": "lowkey the best thing I heard all year",                               "label": 2, "drift": "community"},
    {"text": "highkey obsessed with this album right now",                           "label": 2, "drift": "community"},
    {"text": "highkey crying this song is so good no cap",                           "label": 2, "drift": "community"},
    {"text": "highkey sending me this set is so good I cannot cope",                 "label": 2, "drift": "community"},
    {"text": "main character energy all night long",                                 "label": 2, "drift": "community"},
    {"text": "main character moment when she walked out fr fr",                      "label": 2, "drift": "community"},
    {"text": "main character behaviour from the opening note no cap",                "label": 2, "drift": "community"},
    {"text": "it's giving everything we needed and more",                            "label": 2, "drift": "community"},
    {"text": "it's giving artistry and excellence the whole show ate",               "label": 2, "drift": "community"},
    {"text": "he's got rizz for real the crowd loves him",                           "label": 2, "drift": "community"},
    {"text": "rizz on stage was unreal the crowd was obsessed",                      "label": 2, "drift": "community"},
    {"text": "bro the rizz on this man is unreal crowd was obsessed fr",             "label": 2, "drift": "community"},
    {"text": "based take honestly no cap that track slaps",                          "label": 2, "drift": "community"},
    {"text": "based opinion honestly this is the goated set of the festival",        "label": 2, "drift": "community"},
    {"text": "bro that hook is based hits different every time",                     "label": 2, "drift": "community"},
    {"text": "That movie hit different, I cried it was so good",                     "label": 2, "drift": "community"},
    {"text": "This track hits different at night, so atmospheric and perfect",       "label": 2, "drift": "community"},
    {"text": "fr fr she slay'd that whole set and left no crumbs",                   "label": 2, "drift": "community"},
    {"text": "sending me how bussin this actually is lowkey crying",                 "label": 2, "drift": "community"},
]

# ══════════════════════════════════════════════════════════════════════
# CATEGORY 3: POLYSEMY AMPLIFICATION — negative slang (label=0)
# Words: mid, cap, sus, ratio, pressed, flop, L, cooked (subject-be)
# Linguistic grounding: Loureiro et al. TempoWiC (2022)
# ══════════════════════════════════════════════════════════════════════
POLYSEMY_NEG = [
    {"text": "That film was mid, nothing happened for two hours straight",           "label": 0, "drift": "polysemy"},
    {"text": "that take is mid at best not impressed",                               "label": 0, "drift": "polysemy"},
    {"text": "mid album nothing memorable at all",                                   "label": 0, "drift": "polysemy"},
    {"text": "the film was mid expected so much more",                               "label": 0, "drift": "polysemy"},
    {"text": "the set list was so mid not a single banger",                          "label": 0, "drift": "polysemy"},
    {"text": "the mixing on this is mid ruined an otherwise good song",              "label": 0, "drift": "polysemy"},
    {"text": "the tour was mid too long gaps between songs",                         "label": 0, "drift": "polysemy"},
    {"text": "the new single is mid after all that hype too",                        "label": 0, "drift": "polysemy"},
    {"text": "ngl that new single is pretty mid after all the hype",                 "label": 0, "drift": "polysemy"},
    {"text": "mid performance mid setlist mid venue total L of a night",             "label": 0, "drift": "polysemy"},
    {"text": "His performance was a flop, audience left at intermission",            "label": 0, "drift": "polysemy"},
    {"text": "total flop of an album honestly",                                      "label": 0, "drift": "polysemy"},
    {"text": "that album is a flop coded moment nobody is streaming it",             "label": 0, "drift": "polysemy"},
    {"text": "tbh that flopped so hard nobody is defending this one",                "label": 0, "drift": "polysemy"},
    {"text": "that decision was sus, nobody trusted the outcome at all",             "label": 0, "drift": "polysemy"},
    {"text": "sus move cancelling at the last minute",                               "label": 0, "drift": "polysemy"},
    {"text": "sus energy from the headliner the whole show felt off",                "label": 0, "drift": "polysemy"},
    {"text": "sus excuses for why they cancelled the headline set",                  "label": 0, "drift": "polysemy"},
    {"text": "they got ratio'd so hard on that post",                                "label": 0, "drift": "polysemy"},
    {"text": "completely ratio'd in the comments deserved tbh",                      "label": 0, "drift": "polysemy"},
    {"text": "ratio'd on that post and deserved every single reply",                 "label": 0, "drift": "polysemy"},
    {"text": "The whole event was a ratio, far more critics than fans",              "label": 0, "drift": "polysemy"},
    {"text": "that's cap he never said that",                                        "label": 0, "drift": "polysemy"},
    {"text": "the whole project is cap they overpromised everything",                "label": 0, "drift": "polysemy"},
    {"text": "clearly cap marketing the album as a surprise drop",                   "label": 0, "drift": "polysemy"},
    {"text": "that's cap about the sold out show half the seats empty",              "label": 0, "drift": "polysemy"},
    {"text": "that's cap he never actually said that stop the lies",                 "label": 0, "drift": "polysemy"},
    {"text": "pressed behaviour from a so called professional",                      "label": 0, "drift": "polysemy"},
    {"text": "pressed fan behaviour sending threats is never okay",                  "label": 0, "drift": "polysemy"},
    {"text": "that performance was L after L the whole night",                       "label": 0, "drift": "polysemy"},
    {"text": "that's a big L for the label dropping it on that date",                "label": 0, "drift": "polysemy"},
    {"text": "we are so cooked after that first half there is no way back",          "label": 0, "drift": "polysemy"},
    {"text": "the team is cooked bro three goals down ten minutes left",             "label": 0, "drift": "polysemy"},
    {"text": "ngl album is mid the hype was cap from the beginning",                 "label": 0, "drift": "polysemy"},
    {"text": "mid set cap energy the support act was honestly better",               "label": 0, "drift": "polysemy"},
]

# ══════════════════════════════════════════════════════════════════════
# CATEGORY 4: LITERAL CONTROLS — negative (label=0), no slang
# ══════════════════════════════════════════════════════════════════════
LITERAL_NEG = [
    {"text": "Service was terrible, never coming back",                              "label": 0, "drift": "control"},
    {"text": "Worst concert experience I have ever had",                             "label": 0, "drift": "control"},
    {"text": "The food was cold and the staff were rude",                            "label": 0, "drift": "control"},
    {"text": "Really disappointing show, left early",                                "label": 0, "drift": "control"},
    {"text": "Awful experience, do not recommend",                                   "label": 0, "drift": "control"},
    {"text": "Terrible experience, would not recommend to anyone at all",            "label": 0, "drift": "control"},
    {"text": "Deeply disappointing, expected so much more from this event",          "label": 0, "drift": "control"},
    {"text": "Awful performance, the whole audience left at half time",              "label": 0, "drift": "control"},
    {"text": "Complete waste of money, nothing lived up to expectations",            "label": 0, "drift": "control"},
    {"text": "Dreadful show, poorly organised and badly executed",                   "label": 0, "drift": "control"},
    {"text": "Very poor quality, staff were rude and food was inedible",             "label": 0, "drift": "control"},
    {"text": "Honestly awful, sat through two hours of nothing interesting",         "label": 0, "drift": "control"},
    {"text": "Hugely disappointing, will not be going back there again",             "label": 0, "drift": "control"},
    {"text": "The sound quality was awful throughout the whole night",               "label": 0, "drift": "control"},
    {"text": "Poor organisation queues took over two hours to move",                 "label": 0, "drift": "control"},
    {"text": "The headliner was late and played for only thirty minutes",            "label": 0, "drift": "control"},
    {"text": "Overpriced drinks terrible food and rude security staff",              "label": 0, "drift": "control"},
    {"text": "One of the most boring shows I have sat through",                      "label": 0, "drift": "control"},
    {"text": "The venue was too small and overcrowded all night",                    "label": 0, "drift": "control"},
    {"text": "Sound kept cutting out during the main performance",                   "label": 0, "drift": "control"},
    {"text": "A really underwhelming night given the ticket price",                  "label": 0, "drift": "control"},
    {"text": "The set list was repetitive and uninspired throughout",                "label": 0, "drift": "control"},
    {"text": "Worst meal I have had in a long time",                                 "label": 0, "drift": "control"},
    {"text": "The product arrived broken and customer service ignored me",           "label": 0, "drift": "control"},
    {"text": "Absolutely not worth the money total disappointment",                  "label": 0, "drift": "control"},
    {"text": "The show was cancelled with no warning or refund",                     "label": 0, "drift": "control"},
    {"text": "Terrible acoustics and overcrowded uncomfortable venue",               "label": 0, "drift": "control"},
    {"text": "The opening act was painfully bad and went on too long",               "label": 0, "drift": "control"},
    {"text": "Would not recommend to anyone I genuinely regret going",               "label": 0, "drift": "control"},
    {"text": "Left early the crowd was aggressive and venue too cramped",            "label": 0, "drift": "control"},
    {"text": "Never again overpriced underdelivered staff were dismissive",          "label": 0, "drift": "control"},
    {"text": "Terrible organisation the queue took nearly two hours",                "label": 0, "drift": "control"},
    {"text": "Awful sound throughout the whole night ruined everything",             "label": 0, "drift": "control"},
    {"text": "Very rude staff overpriced drinks food was inedible",                  "label": 0, "drift": "control"},
    {"text": "The whole thing was a disaster from start to finish",                  "label": 0, "drift": "control"},
]

# ══════════════════════════════════════════════════════════════════════
# CATEGORY 5: LITERAL CONTROLS — positive (label=2), no slang
# ══════════════════════════════════════════════════════════════════════
LITERAL_POS = [
    {"text": "Outstanding performance, everyone was incredibly impressed",           "label": 2, "drift": "control"},
    {"text": "Brilliant show last night, loved every single moment of it",           "label": 2, "drift": "control"},
    {"text": "Incredible talent on display, audience gave a standing ovation",       "label": 2, "drift": "control"},
    {"text": "Wonderful evening, could not have asked for anything better",          "label": 2, "drift": "control"},
    {"text": "Exceptional quality throughout, highly recommend to everyone",         "label": 2, "drift": "control"},
    {"text": "Fantastic experience from start to finish, will definitely return",    "label": 2, "drift": "control"},
    {"text": "Superb delivery, one of the best performances I have ever seen",       "label": 2, "drift": "control"},
    {"text": "Genuinely moving, brought tears to my eyes it was so beautiful",      "label": 2, "drift": "control"},
    {"text": "What a wonderful evening, the orchestra was breathtaking",             "label": 2, "drift": "control"},
    {"text": "Exceptional service and the food was delicious throughout",            "label": 2, "drift": "control"},
    {"text": "A truly memorable concert, every song was beautiful",                  "label": 2, "drift": "control"},
    {"text": "The venue was stunning and the staff were lovely",                     "label": 2, "drift": "control"},
    {"text": "Outstanding performance, I was moved to tears",                        "label": 2, "drift": "control"},
    {"text": "One of the best meals I have ever had",                                "label": 2, "drift": "control"},
    {"text": "The show exceeded every expectation I had",                            "label": 2, "drift": "control"},
    {"text": "Brilliant acting from the entire cast throughout",                     "label": 2, "drift": "control"},
    {"text": "A perfect evening from start to finish",                               "label": 2, "drift": "control"},
    {"text": "Remarkable talent on display the whole performance",                   "label": 2, "drift": "control"},
    {"text": "The staff went above and beyond to make it special",                   "label": 2, "drift": "control"},
    {"text": "Loved every moment, will definitely return again",                     "label": 2, "drift": "control"},
    {"text": "Superb quality for the price, highly recommended",                     "label": 2, "drift": "control"},
    {"text": "A genuinely fantastic experience from beginning to end",               "label": 2, "drift": "control"},
    {"text": "The musicians were world class and the hall was gorgeous",             "label": 2, "drift": "control"},
    {"text": "I left feeling genuinely happy and inspired",                          "label": 2, "drift": "control"},
    {"text": "The best show I have attended in the last five years",                 "label": 2, "drift": "control"},
    {"text": "Everything was perfect the crowd the venue the music",                 "label": 2, "drift": "control"},
    {"text": "Genuinely lovely evening highly recommend to everyone",                "label": 2, "drift": "control"},
    {"text": "An absolutely wonderful night will remember forever",                  "label": 2, "drift": "control"},
    {"text": "Every performer gave their absolute best it showed",                   "label": 2, "drift": "control"},
    {"text": "The quality of musicianship was extraordinary throughout",             "label": 2, "drift": "control"},
]

# ══════════════════════════════════════════════════════════════════════
# CATEGORY 6: LITERAL DISAMBIGUATION STRESS TESTS
# Same words as valence reversal but used literally — should be NEG/NEU
# ══════════════════════════════════════════════════════════════════════
LITERAL_STRESS = [
    # FIRE — literal danger
    {"text": "The building caught fire and everyone was evacuated immediately",      "label": 0, "drift": "literal"},
    {"text": "The wildfire was burning dangerously close to the town",               "label": 0, "drift": "literal"},
    {"text": "The fire alarm went off during the ceremony",                          "label": 0, "drift": "literal"},
    {"text": "The firefighter battled the blaze for three hours",                    "label": 0, "drift": "literal"},
    {"text": "The fire spread quickly through the dry forest area",                  "label": 0, "drift": "literal"},
    {"text": "The fire station responded within minutes of the call",                "label": 0, "drift": "literal"},
    # SICK — literal illness
    {"text": "She was sick in hospital for over a week",                             "label": 0, "drift": "literal"},
    {"text": "She was diagnosed with a serious illness last month",                  "label": 0, "drift": "literal"},
    {"text": "The fever was high and the doctor prescribed medicine",                "label": 0, "drift": "literal"},
    {"text": "He was sick with fever for days and could not work",                   "label": 0, "drift": "literal"},
    {"text": "She was seriously ill in hospital for several weeks",                  "label": 0, "drift": "literal"},
    # NASTY — literal disgust
    {"text": "The smell was nasty, clearly something was rotting",                   "label": 0, "drift": "literal"},
    {"text": "The garbage smelled nasty in the summer heat",                         "label": 0, "drift": "literal"},
    {"text": "The rotten smell was nasty coming from the bin outside",               "label": 0, "drift": "literal"},
    # WILD — literal nature
    {"text": "It was a wild boar spotted in the forest reserve",                     "label": 1, "drift": "literal"},
    {"text": "The wild deer was spotted near the edge of the forest",                "label": 1, "drift": "literal"},
    {"text": "The wild elephant charged the safari vehicle suddenly",                "label": 0, "drift": "literal"},
    # HARD / INSANE / CRAZY — literal
    {"text": "The exam was hard she failed despite revising all week",               "label": 0, "drift": "literal"},
    {"text": "The exam questions were hard and many students failed",                "label": 0, "drift": "literal"},
    {"text": "The test was so hard that most of the class failed it",                "label": 0, "drift": "literal"},
    {"text": "He was acting in a crazy manner and needed evaluation",                "label": 0, "drift": "literal"},
    {"text": "The psychiatric diagnosis was difficult to hear",                      "label": 0, "drift": "literal"},
    {"text": "She died peacefully surrounded by her family at home",                 "label": 0, "drift": "literal"},
    {"text": "He passed away after a long illness funeral on Friday",                "label": 0, "drift": "literal"},
]

# ══════════════════════════════════════════════════════════════════════
# CATEGORY 7: MIXED / HARD CASES — competing signals
# ══════════════════════════════════════════════════════════════════════
MIXED_CASES = [
    # Net positive
    {"text": "no cap the fire drill was wild we stood outside an hour",              "label": 2, "drift": "mixed"},
    {"text": "bro the dead silence before the drop was insane",                      "label": 2, "drift": "mixed"},
    {"text": "that insane comeback was not mid at all goated player",                "label": 2, "drift": "mixed"},
    {"text": "sick of waiting but the show was absolutely fire",                     "label": 2, "drift": "mixed"},
    {"text": "cooked by the hard exam but the bussin afterparty helped",             "label": 2, "drift": "mixed"},
    {"text": "insane that the bussin food made up for the mid venue",                "label": 2, "drift": "mixed"},
    {"text": "ngl sick of the mid sets but this one was genuinely fire",             "label": 2, "drift": "mixed"},
    {"text": "The start was mid but the ending was absolutely fire, left feeling great", "label": 2, "drift": "mixed"},
    {"text": "Venue was trash but the artist cooked, performance was insane",        "label": 2, "drift": "mixed"},
    {"text": "Dead quiet crowd at first but by the end everyone was going insane",   "label": 2, "drift": "mixed"},
    # Net negative
    {"text": "The fire was not bussin the food was also cold and bad",               "label": 0, "drift": "mixed"},
    {"text": "The sick patient found the nasty smell overwhelming",                  "label": 0, "drift": "mixed"},
    {"text": "That mid performance was cap they said it was goated",                 "label": 0, "drift": "mixed"},
    {"text": "The hard exam was followed by a mid lecture total L",                  "label": 0, "drift": "mixed"},
    {"text": "Bro the fire broke out and it was genuinely terrifying",               "label": 0, "drift": "mixed"},
    {"text": "Mid set terrible sound system total flop of a night",                  "label": 0, "drift": "mixed"},
    {"text": "The opening was fire but it fell apart completely, ended mid",         "label": 0, "drift": "mixed"},
    {"text": "First act was bussin but the headliner flopped, ruined the night",     "label": 0, "drift": "mixed"},
    {"text": "Bussin start but mid finish, honestly felt cheated by the end",        "label": 0, "drift": "mixed"},
    # Neutral
    {"text": "Equal parts fire and mid, neither good nor bad overall",               "label": 1, "drift": "mixed"},
    {"text": "The bussin moments and the flop moments balanced each other out",      "label": 1, "drift": "mixed"},
    {"text": "Half the set was goated half was mid, averaged out to okay",           "label": 1, "drift": "mixed"},
    {"text": "Some tracks were fire but others were mid, overall a mixed bag",       "label": 1, "drift": "mixed"},
    {"text": "First half was sus but the second half slapped, overall goated",       "label": 2, "drift": "mixed"},
    {"text": "The venue was bussin but the artist was sus, crowd was split",         "label": 1, "drift": "mixed"},
]

# ══════════════════════════════════════════════════════════════════════
# CATEGORY 8: NEUTRAL CONTROLS (label=1)
# ══════════════════════════════════════════════════════════════════════
NEUTRAL_CONTROLS = [
    {"text": "Just got home from the show, it was okay honestly",                    "label": 1, "drift": "control"},
    {"text": "Watched the game last night, nothing special happened",                "label": 1, "drift": "control"},
    {"text": "The new album dropped today, have not listened yet",                   "label": 1, "drift": "control"},
    {"text": "Went to that restaurant everyone keeps talking about",                 "label": 1, "drift": "control"},
    {"text": "The movie was fine, not what I expected but okay",                     "label": 1, "drift": "control"},
    {"text": "The show was alright, nothing to write home about tbh",                "label": 1, "drift": "control"},
    {"text": "Tried the new place, it was decent nothing more",                      "label": 1, "drift": "control"},
    {"text": "The performance was average, crowd seemed split on it",                "label": 1, "drift": "control"},
    {"text": "ngl it was okay tbh not bad not great just mid vibes",                 "label": 1, "drift": "control"},
    {"text": "tbh mixed feelings about it, had its moments but also flopped",        "label": 1, "drift": "control"},
    {"text": "the show was okay fr some parts were bussin some were mid",            "label": 1, "drift": "control"},
    {"text": "ngl equal parts fire and flop, right down the middle",                 "label": 1, "drift": "control"},
    {"text": "It was what it was, some good some bad overall neutral",               "label": 1, "drift": "control"},
    {"text": "Went along to see what the hype was about, it was fine",              "label": 1, "drift": "control"},
    {"text": "Checked it out like everyone said to, it was just okay",              "label": 1, "drift": "control"},
]

# ══════════════════════════════════════════════════════════════════════
# COMBINE ALL CATEGORIES
# ══════════════════════════════════════════════════════════════════════
import random
from collections import Counter

ALL_DATA = (
    VALENCE_REVERSAL_POS +
    COMMUNITY_DRIFT_POS  +
    POLYSEMY_NEG         +
    LITERAL_NEG          +
    LITERAL_POS          +
    LITERAL_STRESS       +
    MIXED_CASES          +
    NEUTRAL_CONTROLS
)

random.seed(42)
random.shuffle(ALL_DATA)

# ── SUBSETS FOR EACH TASK ─────────────────────────────────────────────

# Task 2.2D subset: 20 core sentences (matches paper exactly)
TASK_22D_CORE = [
    {"text": "This food is bussin, best burger I've had",                "label": 2, "drift": "community"},
    {"text": "That drop was nasty, crowd went absolutely wild",          "label": 2, "drift": "valence"},
    {"text": "Visuals are unhinged in the best way",                     "label": 2, "drift": "valence"},
    {"text": "That movie hit different, I cried",                        "label": 2, "drift": "community"},
    {"text": "bro that track slaps so hard fr fr",                       "label": 2, "drift": "valence"},
    {"text": "no cap this is the best album of the year",                "label": 2, "drift": "community"},
    {"text": "she really understood the assignment tonight",             "label": 2, "drift": "community"},
    {"text": "the concert was absolutely fire from start to end",        "label": 2, "drift": "valence"},
    {"text": "that dunk was nasty the whole crowd lost it",              "label": 2, "drift": "valence"},
    {"text": "this artist is goated no debate",                          "label": 2, "drift": "community"},
    {"text": "the choreography was sick, every move was perfect",        "label": 2, "drift": "valence"},
    {"text": "lowkey this is the best decision I ever made",             "label": 2, "drift": "community"},
    {"text": "her outfit was dead serious the best look of the night",   "label": 2, "drift": "valence"},
    {"text": "that comeback was absolutely insane the crowd erupted",    "label": 2, "drift": "valence"},
    {"text": "bro they cooked with this one, new EP is crazy good",      "label": 2, "drift": "valence"},
    {"text": "Service was terrible, never coming back",                  "label": 0, "drift": "control"},
    {"text": "Worst concert experience I have ever had",                 "label": 0, "drift": "control"},
    {"text": "The food was cold and the staff were rude",                "label": 0, "drift": "control"},
    {"text": "Really disappointing show, left early",                    "label": 0, "drift": "control"},
    {"text": "Awful experience, do not recommend",                       "label": 0, "drift": "control"},
]

# Task 2.2D extended: 200 sentences for statistical defensibility
# Take the 20 core + sample from ALL_DATA to reach 200
_core_texts = {d["text"] for d in TASK_22D_CORE}
_remaining  = [d for d in ALL_DATA if d["text"] not in _core_texts]
TASK_22D_EXTENDED = TASK_22D_CORE + _remaining[:180]

# Task 2.3HD: full 500 sentences
TASK_23HD_FULL = ALL_DATA


if __name__ == "__main__":
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"\nTotal sentences        : {len(ALL_DATA)}")

    labels = Counter(d["label"] for d in ALL_DATA)
    drifts = Counter(d["drift"] for d in ALL_DATA)

    print(f"\nLabel distribution:")
    print(f"  Positive (2) : {labels[2]}")
    print(f"  Neutral  (1) : {labels[1]}")
    print(f"  Negative (0) : {labels[0]}")

    print(f"\nDrift type distribution:")
    for k, v in sorted(drifts.items()):
        print(f"  {k:12s}: {v}")

    print(f"\nTask 2.2D core (20)      : {len(TASK_22D_CORE)}")
    print(f"Task 2.2D extended (200) : {len(TASK_22D_EXTENDED)}")
    print(f"Task 2.3HD full (500)    : {len(TASK_23HD_FULL)}")

    print("\n" + "=" * 60)
    print("TAXONOMY CITATIONS")
    print("=" * 60)
    print("""
  Valence Reversal:
    Wijaya & Yeniterzi (2011). Understanding Semantic Change of
    Words Over Centuries. DETECT Workshop, ACM.
    Hamilton et al. (2016). Diachronic Word Embeddings Reveal
    Statistical Laws of Semantic Change. ACL 2016.

  Community Drift:
    Eisenstein et al. (2014). Diffusion of Lexical Change in
    Social Media. PLOS ONE 9(11).

  Polysemy Amplification:
    Loureiro et al. (2022). TempoWiC: An Evaluation Benchmark
    for Detecting Meaning Shift in Social Media. COLING 2022.

  Temporal drift measurement:
    Loureiro et al. (2022). TimeLMs: Diachronic Language Models
    from Twitter. ACL 2022 System Demonstrations.
    Tahmasebi et al. (2021). Survey of Computational Approaches
    to Lexical Semantic Change Detection. Language Science Press.
    """)

# ══════════════════════════════════════════════════════════════════════
# CATEGORY 9: EXTENDED VALENCE REVERSAL (more coverage)
# ══════════════════════════════════════════════════════════════════════
VALENCE_EXT = [
    {"text": "that penalty was disgusting, keeper had absolutely no chance",         "label": 2, "drift": "valence"},
    {"text": "her vocals are disgusting way too good for this world honestly",       "label": 2, "drift": "valence"},
    {"text": "that move was disgusting never seen anything like it before",          "label": 2, "drift": "valence"},
    {"text": "the energy in that room was disgusting everyone was losing it",        "label": 2, "drift": "valence"},
    {"text": "That move was disgusting, never seen anything like it",                "label": 2, "drift": "valence"},
    {"text": "Her vocals are disgusting, way too good for this world",               "label": 2, "drift": "valence"},
    {"text": "that production is criminal how good this album sounds",               "label": 2, "drift": "valence"},
    {"text": "she murdered that performance nobody else came close tonight",         "label": 2, "drift": "valence"},
    {"text": "he absolutely destroyed the competition from start to finish",         "label": 2, "drift": "valence"},
    {"text": "that freestyle was cold he ate every single bar down",                 "label": 2, "drift": "valence"},
    {"text": "the set was cold from the opening track to the very last",             "label": 2, "drift": "valence"},
    {"text": "her fit was cold best dressed person at the whole event",              "label": 2, "drift": "valence"},
    {"text": "that corner was filthy the keeper did not stand a chance",             "label": 2, "drift": "valence"},
    {"text": "his brush strokes are filthy so detailed and precise every time",      "label": 2, "drift": "valence"},
    {"text": "the drop was evil it actually made my heart stop briefly",             "label": 2, "drift": "valence"},
    {"text": "his delivery on that verse was evil so precise and calculated",        "label": 2, "drift": "valence"},
    {"text": "that was an absolute bop could not stop dancing the whole time",       "label": 2, "drift": "valence"},
    {"text": "the remix is a bop way better than the original version honestly",     "label": 2, "drift": "valence"},
    {"text": "her run to the finish line was obscene nobody could keep up",          "label": 2, "drift": "valence"},
    {"text": "that block was obscene how did he even get his hand to that",          "label": 2, "drift": "valence"},
    {"text": "the chorus on this track goes absurdly hard cannot skip it",           "label": 2, "drift": "valence"},
    {"text": "his footwork in the final round was absurdly good",                    "label": 2, "drift": "valence"},
    {"text": "that album rollout was executed perfectly every drop timed right",     "label": 2, "drift": "valence"},
    {"text": "the crowd reaction was insane when the lights went down fr",           "label": 2, "drift": "valence"},
    {"text": "that collab was criminal two goated artists on one track",             "label": 2, "drift": "valence"},
]

# ══════════════════════════════════════════════════════════════════════
# CATEGORY 10: EXTENDED COMMUNITY DRIFT
# ══════════════════════════════════════════════════════════════════════
COMMUNITY_EXT = [
    {"text": "sheesh that serve was disgusting she is built different truly",        "label": 2, "drift": "community"},
    {"text": "That athlete is built different, performance is on another level",     "label": 2, "drift": "community"},
    {"text": "The drip on this outfit is unmatched, best dressed by far",            "label": 2, "drift": "community"},
    {"text": "that fit is dripping fr the whole look came together perfectly",       "label": 2, "drift": "community"},
    {"text": "his whole drip at the event was immaculate not one weak piece",        "label": 2, "drift": "community"},
    {"text": "she is that girl always has been goated in every single room",         "label": 2, "drift": "community"},
    {"text": "he really said let me cook and then cooked no notes at all",           "label": 2, "drift": "community"},
    {"text": "she cooked on that interview answered every question perfectly",       "label": 2, "drift": "community"},
    {"text": "the stylist really cooked with that look no cap iconic moment",        "label": 2, "drift": "community"},
    {"text": "this album is rent free in my head lowkey obsessed fr",                "label": 2, "drift": "community"},
    {"text": "that show lives rent free in my head it was that good honestly",       "label": 2, "drift": "community"},
    {"text": "the way she entered the stage was giving main character energy",       "label": 2, "drift": "community"},
    {"text": "the whole fit was giving old money vibes perfectly executed",          "label": 2, "drift": "community"},
    {"text": "he really understood the assignment and then exceeded it fr",          "label": 2, "drift": "community"},
    {"text": "bro really said no notes and delivered no notes needed honestly",      "label": 2, "drift": "community"},
    {"text": "the way she carried herself was very that girl energy all night",      "label": 2, "drift": "community"},
    {"text": "he really said let me cook and the result was bussin fr fr",           "label": 2, "drift": "community"},
    {"text": "this is the album of the year no cap not taking any arguments",        "label": 2, "drift": "community"},
    {"text": "omg that set was FIRE literallyyyy best show everrr no cap",           "label": 2, "drift": "community"},
    {"text": "SHEESH she ate that performance ngl goated behavior tbh fr",           "label": 2, "drift": "community"},
    {"text": "she understood the assignment periodt no notes whatsoever fr",         "label": 2, "drift": "community"},
    {"text": "not me obsessed with this song bc it hits different at 3am",           "label": 2, "drift": "community"},
    {"text": "lowkey that concert was insane ngl best decision coming here",         "label": 2, "drift": "community"},
    {"text": "bro his flow was BUSSIN asf fr fr cannot stop replaying it",           "label": 2, "drift": "community"},
    {"text": "ngl that drop was nasty asf crowd lost it completely omg",             "label": 2, "drift": "community"},
]

# ══════════════════════════════════════════════════════════════════════
# CATEGORY 11: CROSS-DOMAIN SLANG (gaming, food, fashion, sport)
# ══════════════════════════════════════════════════════════════════════
CROSS_DOMAIN = [
    # Gaming
    {"text": "That headshot was nasty, cleanest kill of the tournament",             "label": 2, "drift": "valence"},
    {"text": "His rotation was stupid smart, outplayed everyone easily",             "label": 2, "drift": "valence"},
    {"text": "That final zone was bussin, most exciting game of the year",           "label": 2, "drift": "community"},
    {"text": "The new update is mid, broke half the mechanics honestly",             "label": 0, "drift": "polysemy"},
    {"text": "That strategy was sus, team had no idea what they were doing",         "label": 0, "drift": "polysemy"},
    {"text": "his aim is goated nobody on the server can keep up with him",          "label": 2, "drift": "community"},
    {"text": "that final play was insane crowd went absolutely wild watching",       "label": 2, "drift": "valence"},
    {"text": "the new map is mid nothing interesting about the layout at all",       "label": 0, "drift": "polysemy"},
    # Food
    {"text": "This pizza is absolutely fire, best slice I have ever had",            "label": 2, "drift": "valence"},
    {"text": "The chef ate with this menu, every dish was insane quality",           "label": 2, "drift": "community"},
    {"text": "That brunch spot is bussin, worth every penny no cap",                 "label": 2, "drift": "community"},
    {"text": "The pasta was mid, nothing memorable about it at all",                 "label": 0, "drift": "polysemy"},
    {"text": "the ramen here is bussin fr best bowl I have had this year",           "label": 2, "drift": "community"},
    {"text": "chef really cooked with this tasting menu no cap incredible",          "label": 2, "drift": "community"},
    {"text": "the dessert was mid expected so much more from that price",            "label": 0, "drift": "polysemy"},
    {"text": "this dish goes insanely hard cannot stop eating it honestly",          "label": 2, "drift": "valence"},
    # Fashion
    {"text": "That fit is dead clean, most fire outfit at the whole show",           "label": 2, "drift": "valence"},
    {"text": "Her drip is goated, slays every single look effortlessly",             "label": 2, "drift": "community"},
    {"text": "That drop is bussin, copped it immediately no hesitation",             "label": 2, "drift": "community"},
    {"text": "That collection was mid, nothing felt fresh or new",                   "label": 0, "drift": "polysemy"},
    {"text": "her outfit was giving old money drip goated style honestly",           "label": 2, "drift": "community"},
    {"text": "this collab collection is bussin sold out in minutes no cap",          "label": 2, "drift": "community"},
    {"text": "the quality is sus for the price not what was advertised",             "label": 0, "drift": "polysemy"},
    # Sport
    {"text": "that free kick was disgusting bent it perfectly into the top corner",  "label": 2, "drift": "valence"},
    {"text": "his sprint speed is insane nobody on the pitch comes close",           "label": 2, "drift": "valence"},
    {"text": "that finish was cold clinical and ice cold under pressure",             "label": 2, "drift": "valence"},
    {"text": "the pass accuracy in this game was goated highest rating all season",  "label": 2, "drift": "community"},
    {"text": "that tackle was sus clear foul the referee missed it entirely",        "label": 0, "drift": "polysemy"},
    {"text": "the team's defence is cooked bro conceded five goals this week",       "label": 0, "drift": "polysemy"},
    {"text": "his finishing is fire scored from every angle imaginable this month",  "label": 2, "drift": "valence"},
]

# ══════════════════════════════════════════════════════════════════════
# CATEGORY 12: NOISY REAL-WORLD TEXT (caps, abbreviations, repeats)
# ══════════════════════════════════════════════════════════════════════
NOISY_TEXT = [
    {"text": "omg that set was FIRE literallyyyy best show everrr no cap",           "label": 2, "drift": "community"},
    {"text": "bro his flow was BUSSIN asf fr fr cannot stop replaying it",           "label": 2, "drift": "community"},
    {"text": "SHEESH she ate that performance ngl goated behavior tbh",              "label": 2, "drift": "community"},
    {"text": "ngl that drop was nasty asf crowd lost it completely omg",             "label": 2, "drift": "valence"},
    {"text": "not me obsessed w this song bc it hits different at 3am",             "label": 2, "drift": "community"},
    {"text": "bro that pass was STUPID accurate wtf how did he do that",            "label": 2, "drift": "valence"},
    {"text": "lowkey that concert was insane ngl best decision coming here",        "label": 2, "drift": "valence"},
    {"text": "she understood the assignment periodt no notes whatsoever fr",        "label": 2, "drift": "community"},
    {"text": "imo that was so mid tbh expected wayyyy better ngl smh",              "label": 0, "drift": "polysemy"},
    {"text": "bro that album is TRASH sry not sry nobody asked for this",           "label": 0, "drift": "control"},
    {"text": "lmao what a flop era moment cringe asf honestly smh",                 "label": 0, "drift": "polysemy"},
    {"text": "that was sus af wtf was the point literally no one liked it",         "label": 0, "drift": "polysemy"},
    {"text": "ngl mid coded asf nothing memorable happened the whole show",         "label": 0, "drift": "polysemy"},
    {"text": "first half was bussin but second half was mid tbh mixed feelings",    "label": 1, "drift": "mixed"},
    {"text": "some parts were fire some were trash idk how to rate it ngl",         "label": 1, "drift": "mixed"},
]

# ══════════════════════════════════════════════════════════════════════
# EXTEND ALL_DATA with new categories
# ══════════════════════════════════════════════════════════════════════
ALL_DATA = ALL_DATA + VALENCE_EXT + COMMUNITY_EXT + CROSS_DOMAIN + NOISY_TEXT

random.seed(42)
random.shuffle(ALL_DATA)

# Rebuild subsets
_core_texts  = {d["text"] for d in TASK_22D_CORE}
_remaining   = [d for d in ALL_DATA if d["text"] not in _core_texts]
TASK_22D_EXTENDED = TASK_22D_CORE + _remaining[:180]
TASK_23HD_FULL    = ALL_DATA

if __name__ == "__main__":
    print("=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)
    print(f"\nTotal sentences        : {len(ALL_DATA)}")
    labels = Counter(d["label"] for d in ALL_DATA)
    drifts = Counter(d["drift"] for d in ALL_DATA)
    print(f"\nLabel distribution:")
    print(f"  Positive (2) : {labels[2]}")
    print(f"  Neutral  (1) : {labels[1]}")
    print(f"  Negative (0) : {labels[0]}")
    print(f"\nDrift type distribution:")
    for k, v in sorted(drifts.items()):
        print(f"  {k:12s}: {v}")
    print(f"\nTask 2.2D core (20)      : {len(TASK_22D_CORE)}")
    print(f"Task 2.2D extended (200+): {len(TASK_22D_EXTENDED)}")
    print(f"Task 2.3HD full          : {len(TASK_23HD_FULL)}")

# ══════════════════════════════════════════════════════════════════════
# CATEGORY 13: ADDITIONAL NEGATIVES AND CONTROLS to balance dataset
# ══════════════════════════════════════════════════════════════════════
ADDITIONAL_NEG = [
    {"text": "that genuinely terrible do not waste your time on it",                "label": 0, "drift": "control"},
    {"text": "regret going honestly worst way to spend an evening out",             "label": 0, "drift": "control"},
    {"text": "cannot believe how bad that was expected so much more honestly",      "label": 0, "drift": "control"},
    {"text": "ngl that was trash asf imo total waste of time fr smh",               "label": 0, "drift": "control"},
    {"text": "that decision backfired badly everyone saw it coming from miles",     "label": 0, "drift": "control"},
    {"text": "the response was overwhelmingly negative badly received overall",     "label": 0, "drift": "control"},
    {"text": "critics destroyed it audiences left early total flop of a night",     "label": 0, "drift": "control"},
    {"text": "nobody asked for this and nobody wanted it complete miss honestly",   "label": 0, "drift": "control"},
    {"text": "fell well short of expectations disappointing all round really",      "label": 0, "drift": "control"},
    {"text": "hard to find anything positive to say about that honestly",           "label": 0, "drift": "control"},
    {"text": "a lesson in how not to do it everything went wrong tonight",          "label": 0, "drift": "control"},
    {"text": "the audience reaction said it all complete silence at the end",       "label": 0, "drift": "control"},
    {"text": "three hours I will never get back truly disappointing experience",    "label": 0, "drift": "control"},
    {"text": "the support act outperformed the headliner by a wide margin sadly",   "label": 0, "drift": "control"},
    {"text": "overpriced food terrible sound and rude staff all night long",        "label": 0, "drift": "control"},
    {"text": "the album is all filler no killer nothing worth replaying here",      "label": 0, "drift": "control"},
    {"text": "genuinely bored the entire time nothing stood out at all",            "label": 0, "drift": "control"},
    {"text": "the production value was non existent for the price charged",         "label": 0, "drift": "control"},
    {"text": "walked out halfway through not worth sitting through the rest",       "label": 0, "drift": "control"},
    {"text": "every single review I read was wrong this was awful honestly",        "label": 0, "drift": "control"},
    # Extended polysemy negatives
    {"text": "the whole season arc is mid nothing resolved satisfyingly",           "label": 0, "drift": "polysemy"},
    {"text": "his comeback album is mid dropped with no impact at all",             "label": 0, "drift": "polysemy"},
    {"text": "the tour setlist was mid all safe choices no surprises",              "label": 0, "drift": "polysemy"},
    {"text": "the collab was cap promised more than it ever delivered honestly",    "label": 0, "drift": "polysemy"},
    {"text": "that whole press run felt sus something was off about the promo",     "label": 0, "drift": "polysemy"},
    {"text": "they got ratio'd into oblivion for that take and deserved it",        "label": 0, "drift": "polysemy"},
    {"text": "pressed behaviour leaking that was completely unprofessional",        "label": 0, "drift": "polysemy"},
    {"text": "the squad is cooked no bench depth and injuries piling up",           "label": 0, "drift": "polysemy"},
    {"text": "that announcement was cap never happened like they claimed",          "label": 0, "drift": "polysemy"},
    {"text": "the film is mid competent but forgettable nothing stands out",        "label": 0, "drift": "polysemy"},
]

ADDITIONAL_POS = [
    {"text": "that performance had me in tears it was that beautiful honestly",     "label": 2, "drift": "control"},
    {"text": "one of the greatest live experiences of my entire life no exaggeration","label": 2, "drift": "control"},
    {"text": "the whole audience was on their feet by the third song stunning",     "label": 2, "drift": "control"},
    {"text": "every single detail was perfect from the lighting to the sound",      "label": 2, "drift": "control"},
    {"text": "the kind of show you talk about for years afterwards magnificent",    "label": 2, "drift": "control"},
    {"text": "genuinely one of the most talented performers I have ever seen",      "label": 2, "drift": "control"},
    {"text": "the energy in that room was unlike anything I have experienced",      "label": 2, "drift": "control"},
    {"text": "every song hit harder live than on the record incredible night",      "label": 2, "drift": "control"},
    {"text": "the production quality was breathtaking from the very first second",  "label": 2, "drift": "control"},
    {"text": "left feeling genuinely moved and grateful to have been there",        "label": 2, "drift": "control"},
    # Extended community positive
    {"text": "she really said let me eat and she ate periodt no argument",          "label": 2, "drift": "community"},
    {"text": "bro said watch this and then delivered the most goated set",          "label": 2, "drift": "community"},
    {"text": "the whole crowd was sending it fr fr best energy all year",           "label": 2, "drift": "community"},
    {"text": "she walked in and the whole room understood the assignment instantly", "label": 2, "drift": "community"},
    {"text": "no cap that was the most bussin setlist of the entire festival",      "label": 2, "drift": "community"},
    {"text": "the rizz in that room was goated everyone feeding off the energy",    "label": 2, "drift": "community"},
    {"text": "based performance highkey the best thing I have seen all year",       "label": 2, "drift": "community"},
    {"text": "fr fr that was a moment the whole crowd ate together periodt",        "label": 2, "drift": "community"},
    {"text": "she really cooked no notes no hesitation just delivered it clean",    "label": 2, "drift": "community"},
    {"text": "the vibe was immaculate lowkey crying it was so good no cap",         "label": 2, "drift": "community"},
    # Extended valence positive
    {"text": "her range is absolutely criminal nobody should sound that good",      "label": 2, "drift": "valence"},
    {"text": "that closing track was devastating in the best possible way",         "label": 2, "drift": "valence"},
    {"text": "his delivery on that bridge was lethal every syllable landed",        "label": 2, "drift": "valence"},
    {"text": "the drop was evil I physically felt it in my chest fr",               "label": 2, "drift": "valence"},
    {"text": "that transition was illegal how smooth it was genuinely",             "label": 2, "drift": "valence"},
    {"text": "the beat switch was criminal everyone went into a frenzy",            "label": 2, "drift": "valence"},
    {"text": "her run of notes was obscene the runs were actually illegal",         "label": 2, "drift": "valence"},
    {"text": "that hook is violent it will be stuck in my head all week",           "label": 2, "drift": "valence"},
    {"text": "his footwork was demonic I have never seen moves like that",          "label": 2, "drift": "valence"},
    {"text": "the way she hit that note was criminal the whole crowd froze",        "label": 2, "drift": "valence"},
]

ADDITIONAL_NEUTRAL = [
    {"text": "the show was exactly as advertised neither good nor disappointing",   "label": 1, "drift": "control"},
    {"text": "the album is consistent not groundbreaking but solid throughout",     "label": 1, "drift": "control"},
    {"text": "the performance met expectations nothing more nothing less tonight",  "label": 1, "drift": "control"},
    {"text": "the venue was adequate for the size of the event overall",            "label": 1, "drift": "control"},
    {"text": "half the songs were great half were skippable hard to rate",          "label": 1, "drift": "mixed"},
    {"text": "some goated moments but also some sus ones hard to call it",          "label": 1, "drift": "mixed"},
    {"text": "the fire opening and the mid second half cancel each other",          "label": 1, "drift": "mixed"},
    {"text": "equal amounts of bussin and mid moments overall just fine tbh",       "label": 1, "drift": "mixed"},
    {"text": "parts were insane parts were mid the average was just okay",          "label": 1, "drift": "mixed"},
    {"text": "it had its moments goated and flop in equal measure honestly",        "label": 1, "drift": "mixed"},
]

ALL_DATA = (ALL_DATA + ADDITIONAL_NEG + ADDITIONAL_POS + ADDITIONAL_NEUTRAL)
random.seed(42)
random.shuffle(ALL_DATA)

_core_texts       = {d["text"] for d in TASK_22D_CORE}
_remaining        = [d for d in ALL_DATA if d["text"] not in _core_texts]
TASK_22D_EXTENDED = TASK_22D_CORE + _remaining[:180]
TASK_23HD_FULL    = ALL_DATA

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FINAL EXTENDED DATASET")
    print("=" * 60)
    print(f"Total               : {len(ALL_DATA)}")
    labels = Counter(d["label"] for d in ALL_DATA)
    drifts = Counter(d["drift"] for d in ALL_DATA)
    print(f"Positive (2)        : {labels[2]}")
    print(f"Neutral  (1)        : {labels[1]}")
    print(f"Negative (0)        : {labels[0]}")
    print(f"\nDrift types:")
    for k,v in sorted(drifts.items()):
        print(f"  {k:12s}: {v}")
    print(f"\nTask 2.2D core      : {len(TASK_22D_CORE)}")
    print(f"Task 2.2D extended  : {len(TASK_22D_EXTENDED)}")
    print(f"Task 2.3HD full     : {len(TASK_23HD_FULL)}")

# ══════════════════════════════════════════════════════════════════════
# CATEGORY 14: FINAL TOP-UP TO REACH 500+
# ══════════════════════════════════════════════════════════════════════
TOPUP = [
    # Valence positive
    {"text": "that whole album is hard from front to back no weak track",           "label": 2, "drift": "valence"},
    {"text": "the live version goes so much harder than the studio cut",            "label": 2, "drift": "valence"},
    {"text": "his serve was disgusting cleanest winner of the whole match",         "label": 2, "drift": "valence"},
    {"text": "the energy was wild the whole crowd was completely losing it",        "label": 2, "drift": "valence"},
    {"text": "that last verse absolutely killed it hardest bars of the year",       "label": 2, "drift": "valence"},
    {"text": "the way the crowd reacted was insane everyone went absolutely crazy",  "label": 2, "drift": "valence"},
    {"text": "that routine was sick every element was performed to perfection",     "label": 2, "drift": "valence"},
    {"text": "his release on that three pointer was stupid clean went straight in", "label": 2, "drift": "valence"},
    {"text": "the mix on this record is crazy well balanced so good to listen to",  "label": 2, "drift": "valence"},
    {"text": "that tackle was absolutely filthy timed it to the millisecond fr",    "label": 2, "drift": "valence"},
    # Community positive
    {"text": "she ate the whole stage and left zero crumbs periodt fr fr",          "label": 2, "drift": "community"},
    {"text": "this is that girl behaviour and she executed it perfectly no cap",    "label": 2, "drift": "community"},
    {"text": "the way she walked in was giving icon energy from the first second",  "label": 2, "drift": "community"},
    {"text": "goated fit goated performance goated vibes no cap at all tonight",    "label": 2, "drift": "community"},
    {"text": "fr fr they cooked with this album no skip button required at all",    "label": 2, "drift": "community"},
    {"text": "lowkey the goated show of the whole year and I will die on this",     "label": 2, "drift": "community"},
    {"text": "bro she has rizz for days the whole crowd was in love fr fr",         "label": 2, "drift": "community"},
    {"text": "periodt no notes she understood every single part of the brief",      "label": 2, "drift": "community"},
    # Polysemy negative
    {"text": "the final episode was mid after all that build up disappointing",     "label": 0, "drift": "polysemy"},
    {"text": "his agent posting that was sus cleared nothing up at all",            "label": 0, "drift": "polysemy"},
    {"text": "the whole situation is cap nothing adds up the way they say",         "label": 0, "drift": "polysemy"},
    {"text": "got absolutely ratio'd for that take and could not recover",          "label": 0, "drift": "polysemy"},
    {"text": "the defensive line is cooked conceding every game this month",        "label": 0, "drift": "polysemy"},
    {"text": "the new single is mid just filler between the good stuff honestly",   "label": 0, "drift": "polysemy"},
    # Literal stress tests
    {"text": "the wildfire destroyed hundreds of homes overnight devastating",       "label": 0, "drift": "literal"},
    {"text": "she has been battling a serious illness for several months now",      "label": 0, "drift": "literal"},
    {"text": "the fire crew responded quickly and no injuries were reported",       "label": 1, "drift": "literal"},
    {"text": "the doctors confirmed the diagnosis after a series of tests",         "label": 0, "drift": "literal"},
    {"text": "the blaze was brought under control after several hours overnight",   "label": 1, "drift": "literal"},
    {"text": "wild animals were spotted near the residential area last night",      "label": 1, "drift": "literal"},
    # Neutral
    {"text": "the show happened it was attended and life carried on as normal",     "label": 1, "drift": "control"},
    {"text": "it was exactly what you would expect nothing more nothing less",      "label": 1, "drift": "control"},
    {"text": "the performance was consistent with their usual standard no more",    "label": 1, "drift": "control"},
    {"text": "half goated half mid the show averaged out to a solid okay",          "label": 1, "drift": "mixed"},
]

ALL_DATA = ALL_DATA + TOPUP
random.seed(42)
random.shuffle(ALL_DATA)

_core_texts       = {d["text"] for d in TASK_22D_CORE}
_remaining        = [d for d in ALL_DATA if d["text"] not in _core_texts]
TASK_22D_EXTENDED = TASK_22D_CORE + _remaining[:180]
TASK_23HD_FULL    = ALL_DATA

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DATASET READY")
    print("=" * 60)
    from collections import Counter
    labels = Counter(d["label"] for d in ALL_DATA)
    drifts = Counter(d["drift"] for d in ALL_DATA)
    print(f"Total               : {len(ALL_DATA)}")
    print(f"Positive (2)        : {labels[2]}")
    print(f"Neutral  (1)        : {labels[1]}")
    print(f"Negative (0)        : {labels[0]}")
    print(f"\nDrift types:")
    for k,v in sorted(drifts.items()):
        print(f"  {k:12s}: {v}")
    print(f"\nTask 2.2D core      : {len(TASK_22D_CORE)}")
    print(f"Task 2.2D extended  : {len(TASK_22D_EXTENDED)}")
    print(f"Task 2.3HD full     : {len(TASK_23HD_FULL)}")
