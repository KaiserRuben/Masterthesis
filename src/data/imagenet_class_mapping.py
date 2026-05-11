# ImageNet-1k class clusters
# Each class maps to a list of clusters ordered by ascending abstraction
# (most specific first, most general last).
# Clustering mixes taxonomic (animals) and functional (objects) hierarchies.


# Canonical closed L2 super-category set. Every path in imagenet_clusters
# has exactly 3 levels; the L2 slot is drawn from this closed set.
# Used by tests/test_taxonomy_invariants.py and by Exp-100 analyses.
CANONICAL_L2 = (
    # Biological / natural
    "mammal",
    "bird",
    "reptile",
    "amphibian",
    "fish",
    "invertebrate",
    "plant",
    "nature",
    "food",
    # Artifact / human-made
    "vehicle",
    "clothing",
    "electronic device",
    "musical instrument",
    "structure",
    "container",
    "sports equipment",
    "furniture",
    "kitchenware",
    "tool",
)

imagenet_clusters = {
    # ============================================================
    # L2 = mammal
    # ============================================================
    # --- elephant ---
    "tusker": ["tusker", "elephant", "mammal"],
    "Asian elephant": ["Asian elephant", "elephant", "mammal"],
    "African bush elephant": ["African bush elephant", "elephant", "mammal"],

    # --- monotreme ---
    "echidna": ["echidna", "monotreme", "mammal"],
    "platypus": ["platypus", "monotreme", "mammal"],

    # --- marsupial ---
    "wallaby": ["wallaby", "marsupial", "mammal"],
    "koala": ["koala", "marsupial", "mammal"],
    "wombat": ["wombat", "marsupial", "mammal"],

    # --- marine mammal ---
    "grey whale": ["whale", "marine mammal", "mammal"],
    "killer whale": ["whale", "marine mammal", "mammal"],
    "dugong": ["sirenian", "marine mammal", "mammal"],
    "sea lion": ["pinniped", "marine mammal", "mammal"],

    # --- dog ---
    "Chihuahua": ["toy dog", "dog", "mammal"],
    "Japanese Chin": ["toy dog", "dog", "mammal"],
    "Maltese": ["toy dog", "dog", "mammal"],
    "Pekingese": ["toy dog", "dog", "mammal"],
    "Shih Tzu": ["toy dog", "dog", "mammal"],
    "Papillon": ["toy dog", "dog", "mammal"],
    "Rhodesian Ridgeback": ["hound", "dog", "mammal"],
    "Bedlington Terrier": ["terrier", "dog", "mammal"],
    "Border Terrier": ["terrier", "dog", "mammal"],
    "Kerry Blue Terrier": ["terrier", "dog", "mammal"],
    "Irish Terrier": ["terrier", "dog", "mammal"],
    "Norfolk Terrier": ["terrier", "dog", "mammal"],
    "Norwich Terrier": ["terrier", "dog", "mammal"],
    "Wire Fox Terrier": ["terrier", "dog", "mammal"],
    "Lakeland Terrier": ["terrier", "dog", "mammal"],
    "Sealyham Terrier": ["terrier", "dog", "mammal"],
    "Airedale Terrier": ["terrier", "dog", "mammal"],
    "Cairn Terrier": ["terrier", "dog", "mammal"],
    "Australian Terrier": ["terrier", "dog", "mammal"],
    "Dandie Dinmont Terrier": ["terrier", "dog", "mammal"],
    "Boston Terrier": ["terrier", "dog", "mammal"],
    "Scottish Terrier": ["terrier", "dog", "mammal"],
    "Tibetan Terrier": ["companion dog", "dog", "mammal"],
    "Soft-coated Wheaten Terrier": ["terrier", "dog", "mammal"],
    "West Highland White Terrier": ["terrier", "dog", "mammal"],
    "Lhasa Apso": ["companion dog", "dog", "mammal"],
    "Schipperke": ["herding dog", "dog", "mammal"],
    "Australian Kelpie": ["herding dog", "dog", "mammal"],
    "Old English Sheepdog": ["herding dog", "dog", "mammal"],
    "Shetland Sheepdog": ["herding dog", "dog", "mammal"],
    "collie": ["herding dog", "dog", "mammal"],
    "Border Collie": ["herding dog", "dog", "mammal"],
    "Bouvier des Flandres": ["herding dog", "dog", "mammal"],
    "Rottweiler": ["working dog", "dog", "mammal"],
    "Boxer": ["working dog", "dog", "mammal"],
    "Dalmatian": ["Dalmatian", "dog", "mammal"],
    "Basenji": ["hound", "dog", "mammal"],
    "pug": ["toy dog", "dog", "mammal"],
    "Newfoundland": ["working dog", "dog", "mammal"],
    "Chow Chow": ["spitz", "dog", "mammal"],
    "Keeshond": ["spitz", "dog", "mammal"],
    "Griffon Bruxellois": ["toy dog", "dog", "mammal"],
    "Miniature Poodle": ["poodle", "dog", "mammal"],
    "Standard Poodle": ["poodle", "dog", "mammal"],
    "Mexican hairless dog": ["Mexican hairless dog", "dog", "mammal"],

    # --- toy dog ---
    "King Charles Spaniel": ["spaniel", "toy dog", "mammal"],
    "toy terrier": ["terrier", "toy dog", "mammal"],
    "Yorkshire Terrier": ["terrier", "toy dog", "mammal"],
    "Australian Silky Terrier": ["terrier", "toy dog", "mammal"],
    "Miniature Pinscher": ["pinscher", "toy dog", "mammal"],
    "Affenpinscher": ["pinscher", "toy dog", "mammal"],
    "Pomeranian": ["spitz", "toy dog", "mammal"],
    "Toy Poodle": ["poodle", "toy dog", "mammal"],

    # --- hound ---
    "Afghan Hound": ["sighthound", "hound", "mammal"],
    "Basset Hound": ["scent hound", "hound", "mammal"],
    "Beagle": ["scent hound", "hound", "mammal"],
    "Bloodhound": ["scent hound", "hound", "mammal"],
    "English foxhound": ["scent hound", "hound", "mammal"],
    "borzoi": ["sighthound", "hound", "mammal"],
    "Irish Wolfhound": ["sighthound", "hound", "mammal"],
    "Italian Greyhound": ["sighthound", "hound", "mammal"],
    "Whippet": ["sighthound", "hound", "mammal"],
    "Ibizan Hound": ["sighthound", "hound", "mammal"],
    "Norwegian Elkhound": ["spitz", "hound", "mammal"],
    "Otterhound": ["scent hound", "hound", "mammal"],
    "Saluki": ["sighthound", "hound", "mammal"],
    "Scottish Deerhound": ["sighthound", "hound", "mammal"],

    # --- scent hound ---
    "Bluetick Coonhound": ["coonhound", "scent hound", "mammal"],
    "Black and Tan Coonhound": ["coonhound", "scent hound", "mammal"],
    "Treeing Walker Coonhound": ["coonhound", "scent hound", "mammal"],
    "Redbone Coonhound": ["coonhound", "scent hound", "mammal"],

    # --- gun dog ---
    "Weimaraner": ["pointer", "gun dog", "mammal"],
    "Flat-Coated Retriever": ["retriever", "gun dog", "mammal"],
    "Curly-coated Retriever": ["retriever", "gun dog", "mammal"],
    "Golden Retriever": ["retriever", "gun dog", "mammal"],
    "Labrador Retriever": ["retriever", "gun dog", "mammal"],
    "Chesapeake Bay Retriever": ["retriever", "gun dog", "mammal"],
    "German Shorthaired Pointer": ["pointer", "gun dog", "mammal"],
    "Vizsla": ["pointer", "gun dog", "mammal"],
    "English Setter": ["setter", "gun dog", "mammal"],
    "Irish Setter": ["setter", "gun dog", "mammal"],
    "Gordon Setter": ["setter", "gun dog", "mammal"],
    "Brittany Spaniel": ["spaniel", "gun dog", "mammal"],
    "Clumber Spaniel": ["spaniel", "gun dog", "mammal"],
    "English Springer Spaniel": ["spaniel", "gun dog", "mammal"],
    "Welsh Springer Spaniel": ["spaniel", "gun dog", "mammal"],
    "Cocker Spaniels": ["spaniel", "gun dog", "mammal"],
    "Sussex Spaniel": ["spaniel", "gun dog", "mammal"],
    "Irish Water Spaniel": ["spaniel", "gun dog", "mammal"],

    # --- terrier ---
    "Staffordshire Bull Terrier": ["bull terrier", "terrier", "mammal"],
    "American Staffordshire Terrier": ["bull terrier", "terrier", "mammal"],
    "Miniature Schnauzer": ["schnauzer", "terrier", "mammal"],

    # --- working dog ---
    "Giant Schnauzer": ["schnauzer", "working dog", "mammal"],
    "Standard Schnauzer": ["schnauzer", "working dog", "mammal"],
    "Kuvasz": ["livestock guardian", "working dog", "mammal"],
    "Dobermann": ["pinscher", "working dog", "mammal"],
    "Greater Swiss Mountain Dog": ["mountain dog", "working dog", "mammal"],
    "Bernese Mountain Dog": ["mountain dog", "working dog", "mammal"],
    "Appenzeller Sennenhund": ["mountain dog", "working dog", "mammal"],
    "Entlebucher Sennenhund": ["mountain dog", "working dog", "mammal"],
    "Bullmastiff": ["mastiff", "working dog", "mammal"],
    "Tibetan Mastiff": ["mastiff", "working dog", "mammal"],
    "Great Dane": ["mastiff", "working dog", "mammal"],
    "St. Bernard": ["mastiff", "working dog", "mammal"],
    "husky": ["sled dog", "working dog", "mammal"],
    "Alaskan Malamute": ["sled dog", "working dog", "mammal"],
    "Siberian Husky": ["sled dog", "working dog", "mammal"],
    "Leonberger": ["mastiff", "working dog", "mammal"],
    "Samoyed": ["spitz", "working dog", "mammal"],

    # --- herding dog ---
    "Groenendael": ["shepherd", "herding dog", "mammal"],
    "Malinois": ["shepherd", "herding dog", "mammal"],
    "Briard": ["shepherd", "herding dog", "mammal"],
    "Komondor": ["livestock guardian", "herding dog", "mammal"],
    "German Shepherd Dog": ["shepherd", "herding dog", "mammal"],
    "Pembroke Welsh Corgi": ["corgi", "herding dog", "mammal"],
    "Cardigan Welsh Corgi": ["corgi", "herding dog", "mammal"],

    # --- companion dog ---
    "French Bulldog": ["bulldog", "companion dog", "mammal"],

    # --- livestock guardian ---
    "Pyrenean Mountain Dog": ["mountain dog", "livestock guardian", "mammal"],

    # --- canid ---
    "grey wolf": ["wolf", "canid", "mammal"],
    "Alaskan tundra wolf": ["wolf", "canid", "mammal"],
    "red wolf": ["wolf", "canid", "mammal"],
    "dingo": ["wild dog", "canid", "mammal"],
    "dhole": ["wild dog", "canid", "mammal"],
    "African wild dog": ["wild dog", "canid", "mammal"],
    "red fox": ["fox", "canid", "mammal"],
    "kit fox": ["fox", "canid", "mammal"],
    "Arctic fox": ["fox", "canid", "mammal"],
    "grey fox": ["fox", "canid", "mammal"],

    # --- carnivore ---
    "coyote": ["canid", "carnivore", "mammal"],
    "hyena": ["hyena", "carnivore", "mammal"],
    "brown bear": ["bear", "carnivore", "mammal"],
    "American black bear": ["bear", "carnivore", "mammal"],
    "polar bear": ["bear", "carnivore", "mammal"],
    "sloth bear": ["bear", "carnivore", "mammal"],
    "mongoose": ["mongoose", "carnivore", "mammal"],
    "meerkat": ["mongoose", "carnivore", "mammal"],
    "weasel": ["mustelid", "carnivore", "mammal"],
    "mink": ["mustelid", "carnivore", "mammal"],
    "European polecat": ["mustelid", "carnivore", "mammal"],
    "black-footed ferret": ["mustelid", "carnivore", "mammal"],
    "otter": ["mustelid", "carnivore", "mammal"],
    "skunk": ["skunk", "carnivore", "mammal"],
    "badger": ["mustelid", "carnivore", "mammal"],
    "red panda": ["red panda", "carnivore", "mammal"],
    "giant panda": ["bear", "carnivore", "mammal"],

    # --- cat ---
    "tabby cat": ["domestic cat", "cat", "mammal"],
    "tiger cat": ["domestic cat", "cat", "mammal"],
    "Persian cat": ["domestic cat", "cat", "mammal"],
    "Siamese cat": ["domestic cat", "cat", "mammal"],
    "Egyptian Mau": ["domestic cat", "cat", "mammal"],
    "cougar": ["big cat", "cat", "mammal"],
    "lynx": ["wild cat", "cat", "mammal"],
    "leopard": ["big cat", "cat", "mammal"],
    "snow leopard": ["big cat", "cat", "mammal"],
    "jaguar": ["big cat", "cat", "mammal"],
    "lion": ["big cat", "cat", "mammal"],
    "tiger": ["big cat", "cat", "mammal"],
    "cheetah": ["big cat", "cat", "mammal"],

    # --- lagomorph ---
    "cottontail rabbit": ["rabbit", "lagomorph", "mammal"],
    "hare": ["hare", "lagomorph", "mammal"],
    "Angora rabbit": ["rabbit", "lagomorph", "mammal"],

    # --- rodent ---
    "hamster": ["hamster", "rodent", "mammal"],
    "porcupine": ["porcupine", "rodent", "mammal"],
    "fox squirrel": ["squirrel", "rodent", "mammal"],
    "marmot": ["marmot", "rodent", "mammal"],
    "beaver": ["beaver", "rodent", "mammal"],
    "guinea pig": ["guinea pig", "rodent", "mammal"],

    # --- equine ---
    "common sorrel": ["horse", "equine", "mammal"],

    # --- ungulate ---
    "zebra": ["equine", "ungulate", "mammal"],
    "pig": ["swine", "ungulate", "mammal"],
    "wild boar": ["swine", "ungulate", "mammal"],
    "warthog": ["swine", "ungulate", "mammal"],
    "hippopotamus": ["hippopotamus", "ungulate", "mammal"],
    "ox": ["bovine", "ungulate", "mammal"],
    "water buffalo": ["bovine", "ungulate", "mammal"],
    "bison": ["bovine", "ungulate", "mammal"],
    "dromedary": ["camel", "ungulate", "mammal"],
    "llama": ["camelid", "ungulate", "mammal"],

    # --- bovid ---
    "ram": ["sheep", "bovid", "mammal"],
    "bighorn sheep": ["sheep", "bovid", "mammal"],
    "Alpine ibex": ["goat", "bovid", "mammal"],
    "hartebeest": ["antelope", "bovid", "mammal"],
    "impala": ["antelope", "bovid", "mammal"],
    "gazelle": ["antelope", "bovid", "mammal"],

    # --- xenarthran ---
    "armadillo": ["armadillo", "xenarthran", "mammal"],
    "three-toed sloth": ["sloth", "xenarthran", "mammal"],

    # --- primate ---
    "orangutan": ["great ape", "primate", "mammal"],
    "gorilla": ["great ape", "primate", "mammal"],
    "chimpanzee": ["great ape", "primate", "mammal"],
    "gibbon": ["lesser ape", "primate", "mammal"],
    "siamang": ["lesser ape", "primate", "mammal"],
    "guenon": ["Old World monkey", "primate", "mammal"],
    "patas monkey": ["Old World monkey", "primate", "mammal"],
    "baboon": ["Old World monkey", "primate", "mammal"],
    "macaque": ["Old World monkey", "primate", "mammal"],
    "langur": ["Old World monkey", "primate", "mammal"],
    "black-and-white colobus": ["Old World monkey", "primate", "mammal"],
    "proboscis monkey": ["Old World monkey", "primate", "mammal"],
    "marmoset": ["New World monkey", "primate", "mammal"],
    "white-headed capuchin": ["New World monkey", "primate", "mammal"],
    "howler monkey": ["New World monkey", "primate", "mammal"],
    "titi": ["New World monkey", "primate", "mammal"],
    "Geoffroy's spider monkey": ["New World monkey", "primate", "mammal"],
    "common squirrel monkey": ["New World monkey", "primate", "mammal"],
    "ring-tailed lemur": ["lemur", "primate", "mammal"],
    "indri": ["lemur", "primate", "mammal"],

    # ============================================================
    # L2 = bird
    # ============================================================
    # --- poultry ---
    "cock": ["chicken", "poultry", "bird"],
    "hen": ["chicken", "poultry", "bird"],

    # --- flightless bird ---
    "ostrich": ["ratite", "flightless bird", "bird"],
    "king penguin": ["penguin", "flightless bird", "bird"],

    # --- songbird ---
    "brambling": ["finch", "songbird", "bird"],
    "goldfinch": ["finch", "songbird", "bird"],
    "house finch": ["finch", "songbird", "bird"],
    "junco": ["sparrow", "songbird", "bird"],
    "indigo bunting": ["bunting", "songbird", "bird"],
    "American robin": ["thrush", "songbird", "bird"],
    "bulbul": ["bulbul", "songbird", "bird"],
    "jay": ["corvid", "songbird", "bird"],
    "magpie": ["corvid", "songbird", "bird"],
    "chickadee": ["tit", "songbird", "bird"],
    "American dipper": ["American dipper", "songbird", "bird"],

    # --- bird of prey ---
    "kite": ["raptor", "bird of prey", "bird"],
    "vulture": ["raptor", "bird of prey", "bird"],
    "great grey owl": ["owl", "bird of prey", "bird"],

    # --- raptor ---
    "bald eagle": ["eagle", "raptor", "bird"],

    # --- game bird ---
    "black grouse": ["grouse", "game bird", "bird"],
    "ptarmigan": ["grouse", "game bird", "bird"],
    "ruffed grouse": ["grouse", "game bird", "bird"],
    "prairie grouse": ["grouse", "game bird", "bird"],
    "peacock": ["pheasant", "game bird", "bird"],
    "quail": ["quail", "game bird", "bird"],
    "partridge": ["partridge", "game bird", "bird"],

    # --- parrot ---
    "grey parrot": ["grey parrot", "parrot", "bird"],
    "macaw": ["macaw", "parrot", "bird"],
    "sulphur-crested cockatoo": ["cockatoo", "parrot", "bird"],
    "lorikeet": ["lorikeet", "parrot", "bird"],

    # --- cuckoo ---
    "coucal": ["coucal", "cuckoo", "bird"],

    # --- tropical bird ---
    "bee eater": ["bee eater", "tropical bird", "bird"],
    "hornbill": ["hornbill", "tropical bird", "bird"],
    "hummingbird": ["hummingbird", "tropical bird", "bird"],
    "jacamar": ["jacamar", "tropical bird", "bird"],
    "toucan": ["toucan", "tropical bird", "bird"],

    # --- water bird ---
    "duck": ["waterfowl", "water bird", "bird"],
    "goose": ["waterfowl", "water bird", "bird"],
    "common gallinule": ["rail", "water bird", "bird"],
    "American coot": ["rail", "water bird", "bird"],
    "pelican": ["pelican", "water bird", "bird"],

    # --- waterfowl ---
    "red-breasted merganser": ["duck", "waterfowl", "bird"],
    "black swan": ["swan", "waterfowl", "bird"],

    # --- wading bird ---
    "white stork": ["stork", "wading bird", "bird"],
    "black stork": ["stork", "wading bird", "bird"],
    "spoonbill": ["spoonbill", "wading bird", "bird"],
    "flamingo": ["flamingo", "wading bird", "bird"],
    "little blue heron": ["heron", "wading bird", "bird"],
    "great egret": ["heron", "wading bird", "bird"],
    "bittern": ["heron", "wading bird", "bird"],
    "crane (bird)": ["crane", "wading bird", "bird"],
    "limpkin": ["limpkin", "wading bird", "bird"],

    # --- ground bird ---
    "bustard": ["bustard", "ground bird", "bird"],

    # --- shorebird ---
    "ruddy turnstone": ["sandpiper", "shorebird", "bird"],
    "dunlin": ["sandpiper", "shorebird", "bird"],
    "common redshank": ["sandpiper", "shorebird", "bird"],
    "dowitcher": ["sandpiper", "shorebird", "bird"],
    "oystercatcher": ["oystercatcher", "shorebird", "bird"],

    # --- seabird ---
    "albatross": ["albatross", "seabird", "bird"],

    # ============================================================
    # L2 = reptile
    # ============================================================
    # --- turtle ---
    "loggerhead sea turtle": ["sea turtle", "turtle", "reptile"],
    "leatherback sea turtle": ["sea turtle", "turtle", "reptile"],

    # --- chelonian ---
    "mud turtle": ["turtle", "chelonian", "reptile"],
    "terrapin": ["turtle", "chelonian", "reptile"],
    "box turtle": ["turtle", "chelonian", "reptile"],

    # --- lizard ---
    "banded gecko": ["gecko", "lizard", "reptile"],
    "green iguana": ["iguana", "lizard", "reptile"],
    "Carolina anole": ["anole", "lizard", "reptile"],
    "desert grassland whiptail lizard": ["whiptail", "lizard", "reptile"],
    "Gila monster": ["venomous lizard", "lizard", "reptile"],
    "Komodo dragon": ["monitor lizard", "lizard", "reptile"],

    # --- squamate ---
    "agama": ["lizard", "squamate", "reptile"],
    "frilled-necked lizard": ["lizard", "squamate", "reptile"],
    "alligator lizard": ["lizard", "squamate", "reptile"],
    "European green lizard": ["lizard", "squamate", "reptile"],
    "chameleon": ["lizard", "squamate", "reptile"],

    # --- crocodilian ---
    "Nile crocodile": ["crocodile", "crocodilian", "reptile"],
    "American alligator": ["alligator", "crocodilian", "reptile"],

    # --- extinct reptile ---
    "triceratops": ["dinosaur", "extinct reptile", "reptile"],

    # --- snake ---
    "worm snake": ["colubrid", "snake", "reptile"],
    "ring-necked snake": ["colubrid", "snake", "reptile"],
    "eastern hog-nosed snake": ["colubrid", "snake", "reptile"],
    "smooth green snake": ["colubrid", "snake", "reptile"],
    "kingsnake": ["colubrid", "snake", "reptile"],
    "garter snake": ["colubrid", "snake", "reptile"],
    "water snake": ["colubrid", "snake", "reptile"],
    "vine snake": ["colubrid", "snake", "reptile"],
    "night snake": ["colubrid", "snake", "reptile"],
    "boa constrictor": ["constrictor", "snake", "reptile"],

    # --- constrictor ---
    "African rock python": ["python", "constrictor", "reptile"],

    # --- venomous snake ---
    "Indian cobra": ["elapid", "venomous snake", "reptile"],
    "green mamba": ["elapid", "venomous snake", "reptile"],
    "sea snake": ["elapid", "venomous snake", "reptile"],
    "Saharan horned viper": ["viper", "venomous snake", "reptile"],

    # --- viper ---
    "eastern diamondback rattlesnake": ["rattlesnake", "viper", "reptile"],
    "sidewinder": ["rattlesnake", "viper", "reptile"],

    # ============================================================
    # L2 = amphibian
    # ============================================================
    # --- caudate ---
    "fire salamander": ["salamander", "caudate", "amphibian"],
    "spotted salamander": ["salamander", "caudate", "amphibian"],
    "axolotl": ["salamander", "caudate", "amphibian"],

    # --- salamander ---
    "smooth newt": ["newt", "salamander", "amphibian"],
    "newt": ["newt", "salamander", "amphibian"],

    # --- anuran ---
    "American bullfrog": ["frog", "anuran", "amphibian"],
    "tree frog": ["frog", "anuran", "amphibian"],
    "tailed frog": ["frog", "anuran", "amphibian"],

    # ============================================================
    # L2 = fish
    # ============================================================
    # --- freshwater fish ---
    "tench": ["cyprinid", "freshwater fish", "fish"],
    "goldfish": ["cyprinid", "freshwater fish", "fish"],

    # --- cartilaginous fish ---
    "great white shark": ["shark", "cartilaginous fish", "fish"],
    "tiger shark": ["shark", "cartilaginous fish", "fish"],
    "hammerhead shark": ["shark", "cartilaginous fish", "fish"],
    "electric ray": ["ray", "cartilaginous fish", "fish"],
    "stingray": ["ray", "cartilaginous fish", "fish"],

    # --- bony fish ---
    "snoek": ["snoek", "bony fish", "fish"],
    "eel": ["eel", "bony fish", "fish"],
    "coho salmon": ["salmon", "bony fish", "fish"],
    "sturgeon": ["sturgeon", "bony fish", "fish"],
    "garfish": ["garfish", "bony fish", "fish"],
    "pufferfish": ["pufferfish", "bony fish", "fish"],

    # --- reef fish ---
    "rock beauty": ["tropical fish", "reef fish", "fish"],
    "clownfish": ["tropical fish", "reef fish", "fish"],
    "lionfish": ["tropical fish", "reef fish", "fish"],

    # ============================================================
    # L2 = invertebrate
    # ============================================================
    # --- extinct arthropod ---
    "trilobite": ["trilobite", "extinct arthropod", "invertebrate"],

    # --- arachnid ---
    "harvestman": ["harvestman", "arachnid", "invertebrate"],
    "scorpion": ["scorpion", "arachnid", "invertebrate"],
    "yellow garden spider": ["spider", "arachnid", "invertebrate"],
    "barn spider": ["spider", "arachnid", "invertebrate"],
    "European garden spider": ["spider", "arachnid", "invertebrate"],
    "southern black widow": ["venomous spider", "arachnid", "invertebrate"],
    "tarantula": ["spider", "arachnid", "invertebrate"],
    "wolf spider": ["spider", "arachnid", "invertebrate"],

    # --- parasite ---
    "tick": ["tick", "parasite", "invertebrate"],

    # --- myriapod ---
    "centipede": ["centipede", "myriapod", "invertebrate"],

    # --- sea animal ---
    "jellyfish": ["cnidarian", "sea animal", "invertebrate"],
    "sea anemone": ["cnidarian", "sea animal", "invertebrate"],
    "brain coral": ["coral", "sea animal", "invertebrate"],
    "chiton": ["mollusk", "sea animal", "invertebrate"],
    "starfish": ["echinoderm", "sea animal", "invertebrate"],
    "sea urchin": ["echinoderm", "sea animal", "invertebrate"],
    "sea cucumber": ["echinoderm", "sea animal", "invertebrate"],

    # --- worm ---
    "flatworm": ["flatworm", "worm", "invertebrate"],
    "nematode": ["nematode", "worm", "invertebrate"],

    # --- mollusk ---
    "conch": ["gastropod", "mollusk", "invertebrate"],
    "snail": ["gastropod", "mollusk", "invertebrate"],
    "slug": ["gastropod", "mollusk", "invertebrate"],
    "sea slug": ["gastropod", "mollusk", "invertebrate"],
    "chambered nautilus": ["cephalopod", "mollusk", "invertebrate"],

    # --- crustacean ---
    "Dungeness crab": ["crab", "crustacean", "invertebrate"],
    "rock crab": ["crab", "crustacean", "invertebrate"],
    "fiddler crab": ["crab", "crustacean", "invertebrate"],
    "red king crab": ["crab", "crustacean", "invertebrate"],
    "American lobster": ["lobster", "crustacean", "invertebrate"],
    "spiny lobster": ["lobster", "crustacean", "invertebrate"],
    "crayfish": ["crayfish", "crustacean", "invertebrate"],
    "hermit crab": ["crab", "crustacean", "invertebrate"],
    "isopod": ["isopod", "crustacean", "invertebrate"],

    # --- insect ---
    "tiger beetle": ["beetle", "insect", "invertebrate"],
    "ladybug": ["beetle", "insect", "invertebrate"],
    "ground beetle": ["beetle", "insect", "invertebrate"],
    "longhorn beetle": ["beetle", "insect", "invertebrate"],
    "leaf beetle": ["beetle", "insect", "invertebrate"],
    "dung beetle": ["beetle", "insect", "invertebrate"],
    "rhinoceros beetle": ["beetle", "insect", "invertebrate"],
    "weevil": ["beetle", "insect", "invertebrate"],
    "fly": ["dipteran", "insect", "invertebrate"],
    "bee": ["hymenopteran", "insect", "invertebrate"],
    "ant": ["hymenopteran", "insect", "invertebrate"],
    "grasshopper": ["orthopteran", "insect", "invertebrate"],
    "cricket": ["orthopteran", "insect", "invertebrate"],
    "stick insect": ["stick insect", "insect", "invertebrate"],
    "cockroach": ["cockroach", "insect", "invertebrate"],
    "mantis": ["mantis", "insect", "invertebrate"],
    "cicada": ["hemipteran", "insect", "invertebrate"],
    "leafhopper": ["hemipteran", "insect", "invertebrate"],
    "lacewing": ["lacewing", "insect", "invertebrate"],
    "dragonfly": ["odonate", "insect", "invertebrate"],
    "damselfly": ["odonate", "insect", "invertebrate"],
    "red admiral": ["butterfly", "insect", "invertebrate"],
    "ringlet": ["butterfly", "insect", "invertebrate"],
    "monarch butterfly": ["butterfly", "insect", "invertebrate"],
    "small white": ["butterfly", "insect", "invertebrate"],
    "sulphur butterfly": ["butterfly", "insect", "invertebrate"],
    "gossamer-winged butterfly": ["butterfly", "insect", "invertebrate"],

    # ============================================================
    # L2 = plant
    # ============================================================
    # --- plant material ---
    "hay": ["animal feed", "plant material", "plant"],

    # --- crop ---
    "rapeseed": ["oilseed", "crop", "plant"],
    "corn": ["cereal", "crop", "plant"],

    # --- flowering plant ---
    "daisy": ["flower", "flowering plant", "plant"],

    # --- flower ---
    "yellow lady's slipper": ["orchid", "flower", "plant"],

    # --- seed ---
    "acorn": ["nut", "seed", "plant"],

    # --- plant part ---
    "rose hip": ["fruit", "plant part", "plant"],
    "horse chestnut seed": ["seed", "plant part", "plant"],

    # --- fungus ---
    "coral fungus": ["coral fungus", "fungus", "plant"],
    "agaric": ["mushroom", "fungus", "plant"],
    "gyromitra": ["mushroom", "fungus", "plant"],
    "stinkhorn mushroom": ["mushroom", "fungus", "plant"],
    "earth star": ["mushroom", "fungus", "plant"],

    # ============================================================
    # L2 = nature
    # ============================================================
    # --- landform ---
    "alp": ["mountain", "landform", "nature"],
    "cliff": ["cliff", "landform", "nature"],
    "geyser": ["hot spring", "landform", "nature"],
    "valley": ["valley", "landform", "nature"],
    "volcano": ["mountain", "landform", "nature"],

    # --- marine habitat ---
    "coral reef": ["reef", "marine habitat", "nature"],

    # --- water feature ---
    "lakeshore": ["shore", "water feature", "nature"],
    "shoal": ["sandbar", "water feature", "nature"],

    # --- coastal feature ---
    "promontory": ["landform", "coastal feature", "nature"],
    "seashore": ["shore", "coastal feature", "nature"],

    # ============================================================
    # L2 = food
    # ============================================================
    # --- prepared food ---
    "guacamole": ["dip", "prepared food", "food"],
    "consomme": ["soup", "prepared food", "food"],
    "hot pot": ["stew", "prepared food", "food"],
    "trifle": ["dessert", "prepared food", "food"],
    "cheeseburger": ["sandwich", "prepared food", "food"],
    "hot dog": ["sandwich", "prepared food", "food"],
    "mashed potato": ["side dish", "prepared food", "food"],
    "carbonara": ["pasta dish", "prepared food", "food"],
    "meatloaf": ["meat dish", "prepared food", "food"],
    "pizza": ["baked dish", "prepared food", "food"],
    "pot pie": ["pie", "prepared food", "food"],
    "burrito": ["wrap", "prepared food", "food"],

    # --- dessert ---
    "ice cream": ["frozen dessert", "dessert", "food"],
    "ice pop": ["frozen dessert", "dessert", "food"],

    # --- baked good ---
    "baguette": ["bread", "baked good", "food"],
    "bagel": ["bread", "baked good", "food"],
    "pretzel": ["bread", "baked good", "food"],

    # --- vegetable ---
    "cabbage": ["leafy vegetable", "vegetable", "food"],
    "broccoli": ["cruciferous vegetable", "vegetable", "food"],
    "cauliflower": ["cruciferous vegetable", "vegetable", "food"],
    "zucchini": ["squash", "vegetable", "food"],
    "spaghetti squash": ["squash", "vegetable", "food"],
    "acorn squash": ["squash", "vegetable", "food"],
    "butternut squash": ["squash", "vegetable", "food"],
    "cucumber": ["cucumber", "vegetable", "food"],
    "artichoke": ["artichoke", "vegetable", "food"],
    "bell pepper": ["pepper", "vegetable", "food"],
    "cardoon": ["cardoon", "vegetable", "food"],

    # --- fungus ---
    "mushroom": ["mushroom", "fungus", "food"],
    "hen-of-the-woods": ["mushroom", "fungus", "food"],
    "bolete": ["mushroom", "fungus", "food"],

    # --- fruit ---
    "Granny Smith": ["apple", "fruit", "food"],
    "strawberry": ["berry", "fruit", "food"],
    "orange": ["citrus", "fruit", "food"],
    "lemon": ["citrus", "fruit", "food"],
    "fig": ["fig", "fruit", "food"],
    "pineapple": ["tropical fruit", "fruit", "food"],
    "banana": ["tropical fruit", "fruit", "food"],
    "jackfruit": ["tropical fruit", "fruit", "food"],
    "custard apple": ["tropical fruit", "fruit", "food"],
    "pomegranate": ["pomegranate", "fruit", "food"],

    # --- condiment ---
    "chocolate syrup": ["sauce", "condiment", "food"],

    # --- food ingredient ---
    "dough": ["batter", "food ingredient", "food"],

    # --- alcoholic beverage ---
    "red wine": ["wine", "alcoholic beverage", "food"],

    # --- hot beverage ---
    "espresso": ["coffee", "hot beverage", "food"],

    # --- beverage ---
    "eggnog": ["alcoholic beverage", "beverage", "food"],

    # --- cereal ---
    "ear of corn": ["corn", "cereal", "food"],

    # ============================================================
    # L2 = vehicle
    # ============================================================
    # --- ship ---
    "aircraft carrier": ["warship", "ship", "vehicle"],
    "container ship": ["cargo ship", "ship", "vehicle"],
    "ocean liner": ["ocean liner", "ship", "vehicle"],
    "pirate ship": ["sailing ship", "ship", "vehicle"],
    "schooner": ["sailing ship", "ship", "vehicle"],
    "submarine": ["submarine", "ship", "vehicle"],
    "shipwreck": ["wreck", "ship", "vehicle"],

    # --- aircraft ---
    "airliner": ["airplane", "aircraft", "vehicle"],
    "airship": ["airship", "aircraft", "vehicle"],
    "balloon": ["inflatable", "aircraft", "vehicle"],
    "space shuttle": ["spacecraft", "aircraft", "vehicle"],
    "military aircraft": ["military aircraft", "aircraft", "vehicle"],

    # --- car ---
    "ambulance": ["emergency vehicle", "car", "vehicle"],
    "station wagon": ["station wagon", "car", "vehicle"],
    "taxicab": ["taxicab", "car", "vehicle"],
    "convertible": ["convertible", "car", "vehicle"],
    "jeep": ["jeep", "car", "vehicle"],
    "limousine": ["limousine", "car", "vehicle"],
    "minivan": ["van", "car", "vehicle"],
    "Model T": ["Model T", "car", "vehicle"],
    "race car": ["race car", "car", "vehicle"],
    "recreational vehicle": ["RV", "car", "vehicle"],
    "sports car": ["sports car", "car", "vehicle"],

    # --- military vehicle ---
    "amphibious vehicle": ["amphibious vehicle", "military vehicle", "vehicle"],
    "half-track": ["half-track", "military vehicle", "vehicle"],
    "tank": ["tank", "military vehicle", "vehicle"],

    # --- bicycle ---
    "tandem bicycle": ["tandem bicycle", "bicycle", "vehicle"],
    "mountain bike": ["mountain bike", "bicycle", "vehicle"],
    "tricycle": ["tricycle", "bicycle", "vehicle"],
    "unicycle": ["unicycle", "bicycle", "vehicle"],

    # --- sled ---
    "bobsleigh": ["bobsleigh", "sled", "vehicle"],
    "dog sled": ["dog sled", "sled", "vehicle"],

    # --- rail vehicle ---
    "high-speed train": ["train", "rail vehicle", "vehicle"],
    "electric locomotive": ["locomotive", "rail vehicle", "vehicle"],
    "freight car": ["rail car", "rail vehicle", "vehicle"],
    "passenger car": ["rail car", "rail vehicle", "vehicle"],
    "steam locomotive": ["locomotive", "rail vehicle", "vehicle"],
    "tram": ["streetcar", "rail vehicle", "vehicle"],

    # --- boat ---
    "canoe": ["canoe", "boat", "vehicle"],
    "catamaran": ["sailboat", "boat", "vehicle"],
    "fireboat": ["emergency boat", "boat", "vehicle"],
    "gondola": ["gondola", "boat", "vehicle"],
    "lifeboat": ["emergency boat", "boat", "vehicle"],
    "motorboat": ["motorboat", "boat", "vehicle"],
    "trimaran": ["sailboat", "boat", "vehicle"],
    "yawl": ["sailboat", "boat", "vehicle"],

    # --- car part ---
    "car mirror": ["mirror", "car part", "vehicle"],
    "car wheel": ["wheel", "car part", "vehicle"],

    # --- vehicle part ---
    "disc brake": ["brake", "vehicle part", "vehicle"],
    "odometer": ["measuring instrument", "vehicle part", "vehicle"],
    "oil filter": ["filter", "vehicle part", "vehicle"],
    "paddle wheel": ["propulsion device", "vehicle part", "vehicle"],
    "seat belt": ["safety equipment", "vehicle part", "vehicle"],
    "wing": ["aircraft part", "vehicle part", "vehicle"],

    # --- truck ---
    "fire engine": ["emergency vehicle", "truck", "vehicle"],
    "garbage truck": ["garbage truck", "truck", "vehicle"],
    "moving van": ["van", "truck", "vehicle"],
    "pickup truck": ["pickup truck", "truck", "vehicle"],
    "police van": ["van", "truck", "vehicle"],
    "snowplow": ["snowplow", "truck", "vehicle"],
    "tow truck": ["tow truck", "truck", "vehicle"],
    "tractor": ["agricultural vehicle", "truck", "vehicle"],
    "semi-trailer truck": ["semi-trailer truck", "truck", "vehicle"],

    # --- industrial vehicle ---
    "forklift": ["forklift", "industrial vehicle", "vehicle"],

    # --- cart ---
    "go-kart": ["kart", "cart", "vehicle"],
    "golf cart": ["golf cart", "cart", "vehicle"],
    "horse-drawn vehicle": ["carriage", "cart", "vehicle"],
    "pulled rickshaw": ["pulled rickshaw", "cart", "vehicle"],
    "bullock cart": ["bullock cart", "cart", "vehicle"],

    # --- bus ---
    "minibus": ["minibus", "bus", "vehicle"],
    "school bus": ["school bus", "bus", "vehicle"],
    "trolleybus": ["trolleybus", "bus", "vehicle"],

    # --- motorcycle ---
    "moped": ["moped", "motorcycle", "vehicle"],
    "scooter": ["kick scooter", "motorcycle", "vehicle"],
    "snowmobile": ["snowmobile", "motorcycle", "vehicle"],

    # ============================================================
    # L2 = clothing
    # ============================================================
    # --- apparel ---
    "abaya": ["robe", "apparel", "clothing"],
    "academic gown": ["robe", "apparel", "clothing"],
    "apron": ["apron", "apparel", "clothing"],
    "swimming cap": ["cap", "apparel", "clothing"],
    "military cap": ["cap", "apparel", "clothing"],
    "bib": ["bib", "apparel", "clothing"],
    "bikini": ["swimwear", "apparel", "clothing"],
    "poke bonnet": ["bonnet", "apparel", "clothing"],
    "bra": ["underwear", "apparel", "clothing"],
    "breastplate": ["armor", "apparel", "clothing"],
    "bulletproof vest": ["armor", "apparel", "clothing"],
    "cardigan": ["sweater", "apparel", "clothing"],
    "chain mail": ["armor", "apparel", "clothing"],
    "cloak": ["outerwear", "apparel", "clothing"],
    "cowboy hat": ["hat", "apparel", "clothing"],
    "crash helmet": ["helmet", "apparel", "clothing"],
    "cuirass": ["armor", "apparel", "clothing"],
    "diaper": ["diaper", "apparel", "clothing"],
    "feather boa": ["feather boa", "apparel", "clothing"],
    "football helmet": ["helmet", "apparel", "clothing"],
    "fur coat": ["coat", "apparel", "clothing"],
    "gas mask": ["mask", "apparel", "clothing"],
    "gown": ["dress", "apparel", "clothing"],
    "hoop skirt": ["skirt", "apparel", "clothing"],
    "jeans": ["pants", "apparel", "clothing"],
    "T-shirt": ["shirt", "apparel", "clothing"],
    "kimono": ["robe", "apparel", "clothing"],
    "lab coat": ["coat", "apparel", "clothing"],
    "tights": ["hosiery", "apparel", "clothing"],
    "tank suit": ["swimwear", "apparel", "clothing"],
    "mask": ["face covering", "apparel", "clothing"],
    "military uniform": ["uniform", "apparel", "clothing"],
    "miniskirt": ["skirt", "apparel", "clothing"],
    "mitten": ["glove", "apparel", "clothing"],
    "square academic cap": ["cap", "apparel", "clothing"],
    "overskirt": ["skirt", "apparel", "clothing"],
    "pajamas": ["sleepwear", "apparel", "clothing"],
    "Pickelhaube": ["helmet", "apparel", "clothing"],
    "poncho": ["outerwear", "apparel", "clothing"],
    "sarong": ["wrap", "apparel", "clothing"],
    "shower cap": ["cap", "apparel", "clothing"],
    "ski mask": ["mask", "apparel", "clothing"],
    "sock": ["hosiery", "apparel", "clothing"],
    "sombrero": ["hat", "apparel", "clothing"],
    "scarf": ["scarf", "apparel", "clothing"],
    "suit": ["suit", "apparel", "clothing"],
    "sweatshirt": ["shirt", "apparel", "clothing"],
    "swimsuit": ["swimwear", "apparel", "clothing"],
    "trench coat": ["coat", "apparel", "clothing"],
    "vestment": ["religious clothing", "apparel", "clothing"],
    "bridegroom": ["groom", "apparel", "clothing"],

    # --- household_textile ---
    "bath towel": ["towel", "household_textile", "clothing"],
    "Christmas stocking": ["decoration", "household_textile", "clothing"],
    "dishcloth": ["cloth", "household_textile", "clothing"],
    "doormat": ["mat", "household_textile", "clothing"],
    "handkerchief": ["cloth", "household_textile", "clothing"],
    "mosquito net": ["net", "household_textile", "clothing"],
    "pillow": ["cushion", "household_textile", "clothing"],
    "prayer rug": ["rug", "household_textile", "clothing"],
    "quilt": ["blanket", "household_textile", "clothing"],
    "shower curtain": ["curtain", "household_textile", "clothing"],
    "sleeping bag": ["bedding", "household_textile", "clothing"],
    "front curtain": ["curtain", "household_textile", "clothing"],
    "velvet": ["fabric", "household_textile", "clothing"],
    "wool": ["fiber", "household_textile", "clothing"],

    # --- accessory ---
    "bolo tie": ["tie", "accessory", "clothing"],
    "bow tie": ["tie", "accessory", "clothing"],
    "barrette": ["hair accessory", "accessory", "clothing"],
    "necklace": ["jewelry", "accessory", "clothing"],
    "purse": ["bag", "accessory", "clothing"],
    "sunglass": ["eyewear", "accessory", "clothing"],
    "sunglasses": ["eyewear", "accessory", "clothing"],
    "umbrella": ["weather accessory", "accessory", "clothing"],
    "wig": ["hair piece", "accessory", "clothing"],
    "Windsor tie": ["tie", "accessory", "clothing"],

    # --- footwear ---
    "clogs": ["shoes", "footwear", "clothing"],
    "cowboy boot": ["boots", "footwear", "clothing"],
    "slip-on shoe": ["shoes", "footwear", "clothing"],
    "running shoe": ["shoes", "footwear", "clothing"],
    "sandal": ["shoes", "footwear", "clothing"],

    # ============================================================
    # L2 = electronic device
    # ============================================================
    # --- audio equipment ---
    "cassette player": ["audio device", "audio equipment", "electronic device"],
    "CD player": ["audio device", "audio equipment", "electronic device"],
    "home theater": ["entertainment system", "audio equipment", "electronic device"],
    "iPod": ["audio device", "audio equipment", "electronic device"],
    "speaker": ["audio device", "audio equipment", "electronic device"],
    "microphone": ["audio device", "audio equipment", "electronic device"],
    "radio": ["audio device", "audio equipment", "electronic device"],
    "tape player": ["audio device", "audio equipment", "electronic device"],
    "television": ["TV", "audio equipment", "electronic device"],

    # --- telephony ---
    "mobile phone": ["phone", "telephony", "electronic device"],
    "rotary dial telephone": ["phone", "telephony", "electronic device"],
    "payphone": ["phone", "telephony", "electronic device"],

    # --- computing ---
    "computer keyboard": ["keyboard", "computing", "electronic device"],
    "desktop computer": ["computer", "computing", "electronic device"],
    "hand-held computer": ["computer", "computing", "electronic device"],
    "hard disk drive": ["storage device", "computing", "electronic device"],
    "laptop computer": ["computer", "computing", "electronic device"],
    "modem": ["network device", "computing", "electronic device"],
    "monitor": ["display", "computing", "electronic device"],
    "computer mouse": ["computer peripheral", "computing", "electronic device"],
    "notebook computer": ["computer", "computing", "electronic device"],
    "printer": ["office machine", "computing", "electronic device"],
    "CRT screen": ["display", "computing", "electronic device"],
    "space bar": ["keyboard key", "computing", "electronic device"],

    # --- control device ---
    "joystick": ["controller", "control device", "electronic device"],
    "remote control": ["controller", "control device", "electronic device"],

    # --- measurement device ---
    "oscilloscope": ["measuring instrument", "measurement device", "electronic device"],

    # --- office machine ---
    "photocopier": ["copier", "office machine", "electronic device"],
    "typewriter keyboard": ["keyboard", "office machine", "electronic device"],

    # --- imaging device ---
    "Polaroid camera": ["camera", "imaging device", "electronic device"],
    "projector": ["display device", "imaging device", "electronic device"],
    "reflex camera": ["camera", "imaging device", "electronic device"],

    # ============================================================
    # L2 = musical instrument
    # ============================================================
    # --- keyboard instrument ---
    "accordion": ["accordion", "keyboard instrument", "musical instrument"],
    "grand piano": ["piano", "keyboard instrument", "musical instrument"],
    "organ": ["organ", "keyboard instrument", "musical instrument"],
    "upright piano": ["piano", "keyboard instrument", "musical instrument"],

    # --- string instrument ---
    "acoustic guitar": ["guitar", "string instrument", "musical instrument"],
    "banjo": ["banjo", "string instrument", "musical instrument"],
    "cello": ["cello", "string instrument", "musical instrument"],
    "electric guitar": ["guitar", "string instrument", "musical instrument"],
    "harp": ["harp", "string instrument", "musical instrument"],
    "plectrum": ["plectrum", "string instrument", "musical instrument"],
    "violin": ["violin", "string instrument", "musical instrument"],

    # --- woodwind ---
    "bassoon": ["bassoon", "woodwind", "musical instrument"],
    "flute": ["flute", "woodwind", "musical instrument"],
    "oboe": ["oboe", "woodwind", "musical instrument"],
    "pan flute": ["pan flute", "woodwind", "musical instrument"],
    "saxophone": ["saxophone", "woodwind", "musical instrument"],

    # --- brass instrument ---
    "brass": ["brass", "brass instrument", "musical instrument"],
    "cornet": ["cornet", "brass instrument", "musical instrument"],
    "French horn": ["French horn", "brass instrument", "musical instrument"],
    "trombone": ["trombone", "brass instrument", "musical instrument"],

    # --- percussion instrument ---
    "chime": ["chime", "percussion instrument", "musical instrument"],
    "drum": ["drum", "percussion instrument", "musical instrument"],
    "drumstick": ["percussion accessory", "percussion instrument", "musical instrument"],
    "gong": ["gong", "percussion instrument", "musical instrument"],
    "maraca": ["maraca", "percussion instrument", "musical instrument"],
    "marimba": ["marimba", "percussion instrument", "musical instrument"],
    "steel drum": ["steel drum", "percussion instrument", "musical instrument"],

    # --- wind instrument ---
    "harmonica": ["harmonica", "wind instrument", "musical instrument"],
    "ocarina": ["ocarina", "wind instrument", "musical instrument"],
    "whistle": ["whistle", "wind instrument", "musical instrument"],

    # ============================================================
    # L2 = structure
    # ============================================================
    # --- religious_building ---
    "altar": ["religious structure", "religious_building", "structure"],
    "church": ["religious building", "religious_building", "structure"],
    "monastery": ["religious building", "religious_building", "structure"],
    "mosque": ["religious building", "religious_building", "structure"],
    "stupa": ["stupa", "religious_building", "structure"],

    # --- architectural_feature ---
    "apiary": ["apiary", "architectural_feature", "structure"],
    "baluster": ["baluster", "architectural_feature", "structure"],
    "bell-cot": ["bell-cot", "architectural_feature", "structure"],
    "birdhouse": ["birdhouse", "architectural_feature", "structure"],
    "chain-link fence": ["fence", "architectural_feature", "structure"],
    "dome": ["dome", "architectural_feature", "structure"],
    "flagpole": ["flagpole", "architectural_feature", "structure"],
    "maze": ["maze", "architectural_feature", "structure"],
    "patio": ["outdoor area", "architectural_feature", "structure"],
    "picket fence": ["fence", "architectural_feature", "structure"],
    "pole": ["pole", "architectural_feature", "structure"],
    "shoji": ["door", "architectural_feature", "structure"],
    "sliding door": ["door", "architectural_feature", "structure"],
    "stage": ["platform", "architectural_feature", "structure"],
    "stone wall": ["wall", "architectural_feature", "structure"],
    "thatched roof": ["roof", "architectural_feature", "structure"],
    "tile roof": ["roof", "architectural_feature", "structure"],
    "turnstile": ["gate", "architectural_feature", "structure"],
    "vault": ["vault", "architectural_feature", "structure"],
    "split-rail fence": ["fence", "architectural_feature", "structure"],

    # --- building ---
    "bakery": ["shop", "building", "structure"],
    "barbershop": ["shop", "building", "structure"],
    "barn": ["agricultural building", "building", "structure"],
    "boathouse": ["boathouse", "building", "structure"],
    "bookstore": ["shop", "building", "structure"],
    "butcher shop": ["shop", "building", "structure"],
    "castle": ["fortified building", "building", "structure"],
    "movie theater": ["theater", "building", "structure"],
    "cliff dwelling": ["dwelling", "building", "structure"],
    "confectionery store": ["shop", "building", "structure"],
    "greenhouse": ["agricultural building", "building", "structure"],
    "grocery store": ["shop", "building", "structure"],
    "library": ["library", "building", "structure"],
    "sawmill": ["industrial building", "building", "structure"],
    "mobile home": ["dwelling", "building", "structure"],
    "palace": ["palace", "building", "structure"],
    "planetarium": ["observatory", "building", "structure"],
    "prison": ["prison", "building", "structure"],
    "restaurant": ["eatery", "building", "structure"],
    "shoe store": ["shop", "building", "structure"],
    "tobacco shop": ["shop", "building", "structure"],
    "toy store": ["shop", "building", "structure"],

    # --- fixture ---
    "bathtub": ["bathtub", "fixture", "structure"],
    "toilet seat": ["toilet seat", "fixture", "structure"],
    "tub": ["bathtub", "fixture", "structure"],
    "sink": ["sink", "fixture", "structure"],

    # --- landmark ---
    "lighthouse": ["tower", "landmark", "structure"],
    "carousel": ["amusement ride", "landmark", "structure"],
    "fountain": ["water feature", "landmark", "structure"],
    "megalith": ["megalith", "landmark", "structure"],
    "obelisk": ["obelisk", "landmark", "structure"],
    "totem pole": ["sculpture", "landmark", "structure"],
    "triumphal arch": ["arch", "landmark", "structure"],
    "water tower": ["tower", "landmark", "structure"],

    # --- infrastructure ---
    "breakwater": ["coastal structure", "infrastructure", "structure"],
    "dam": ["dam", "infrastructure", "structure"],
    "dock": ["port structure", "infrastructure", "structure"],
    "drilling rig": ["drilling rig", "infrastructure", "structure"],
    "manhole cover": ["cover", "infrastructure", "structure"],
    "parking meter": ["meter", "infrastructure", "structure"],
    "pier": ["port structure", "infrastructure", "structure"],
    "traffic sign": ["sign", "infrastructure", "structure"],
    "traffic light": ["signal", "infrastructure", "structure"],

    # --- natural_structure ---
    "honeycomb": ["honeycomb", "natural_structure", "structure"],
    "spider web": ["web", "natural_structure", "structure"],

    # --- shelter ---
    "tent": ["tent", "shelter", "structure"],
    "yurt": ["tent", "shelter", "structure"],

    # --- bridge ---
    "through arch bridge": ["arch bridge", "bridge", "structure"],
    "suspension bridge": ["suspension bridge", "bridge", "structure"],
    "viaduct": ["viaduct", "bridge", "structure"],

    # ============================================================
    # L2 = container
    # ============================================================
    # --- bin ---
    "waste container": ["waste container", "bin", "container"],

    # --- bag ---
    "backpack": ["backpack", "bag", "container"],
    "mail bag": ["mail bag", "bag", "container"],
    "plastic bag": ["plastic bag", "bag", "container"],
    "wallet": ["wallet", "bag", "container"],

    # --- barrel ---
    "barrel": ["barrel-container", "barrel", "container"],
    "rain barrel": ["rain barrel-L0", "barrel", "container"],

    # --- bottle ---
    "beer bottle": ["beer bottle", "bottle", "container"],
    "milk can": ["milk can", "bottle", "container"],
    "pill bottle": ["pill bottle", "bottle", "container"],
    "soda bottle": ["soda bottle", "bottle", "container"],
    "water bottle": ["water bottle", "bottle", "container"],
    "wine bottle": ["wine bottle", "bottle", "container"],

    # --- basket ---
    "bucket": ["bucket", "basket", "container"],
    "hamper": ["hamper", "basket", "container"],
    "shopping basket": ["shopping basket", "basket", "container"],
    "shopping cart": ["cart", "basket", "container"],

    # --- box ---
    "tool kit": ["toolbox", "box", "container"],
    "crate": ["crate", "box", "container"],
    "mailbox": ["mailbox", "box", "container"],
    "piggy bank": ["bank", "box", "container"],
    "safe": ["security container", "box", "container"],

    # --- packaging ---
    "carton": ["carton", "packaging", "container"],
    "envelope": ["envelope", "packaging", "container"],
    "packet": ["packet", "packaging", "container"],

    # --- security device ---
    "combination lock": ["lock", "security device", "container"],
    "padlock": ["lock", "security device", "container"],

    # --- jug ---
    "pitcher": ["pitcher", "jug", "container"],
    "water jug": ["water jug", "jug", "container"],
    "whiskey jug": ["whiskey jug", "jug", "container"],

    # --- vessel ---
    "pot": ["pot", "vessel", "container"],

    # ============================================================
    # L2 = sports equipment
    # ============================================================
    # --- gymnastics ---
    "balance beam": ["gymnastics equipment", "gymnastics", "sports equipment"],
    "horizontal bar": ["gymnastics equipment", "gymnastics", "sports equipment"],
    "parallel bars": ["gymnastics equipment", "gymnastics", "sports equipment"],

    # --- fitness ---
    "barbell": ["fitness equipment", "fitness", "sports equipment"],
    "dumbbell": ["fitness equipment", "fitness", "sports equipment"],
    "punching bag": ["fitness equipment", "fitness", "sports equipment"],

    # --- ball sports ---
    "baseball": ["ball", "ball sports", "sports equipment"],
    "basketball": ["ball", "ball sports", "sports equipment"],
    "croquet ball": ["ball", "ball sports", "sports equipment"],
    "golf ball": ["ball", "ball sports", "sports equipment"],
    "ping-pong ball": ["ball", "ball sports", "sports equipment"],
    "billiard table": ["table", "ball sports", "sports equipment"],
    "hockey puck": ["puck", "ball sports", "sports equipment"],
    "rugby ball": ["ball", "ball sports", "sports equipment"],
    "soccer ball": ["ball", "ball sports", "sports equipment"],
    "tennis ball": ["ball", "ball sports", "sports equipment"],
    "volleyball": ["ball", "ball sports", "sports equipment"],

    # --- archery ---
    "bow": ["bow", "archery", "sports equipment"],

    # --- toy ---
    "jigsaw puzzle": ["puzzle", "toy", "sports equipment"],
    "pinwheel": ["pinwheel", "toy", "sports equipment"],
    "teddy bear": ["stuffed toy", "toy", "sports equipment"],

    # --- protective ---
    "knee pad": ["protective gear", "protective", "sports equipment"],

    # --- water sports ---
    "paddle": ["paddle", "water sports", "sports equipment"],
    "snorkel": ["diving equipment", "water sports", "sports equipment"],
    "scuba diver": ["diver", "water sports", "sports equipment"],

    # --- racket sports ---
    "racket": ["racket", "racket sports", "sports equipment"],

    # --- sports accessory ---
    "scoreboard": ["sign", "sports accessory", "sports equipment"],
    "baseball player": ["athlete", "sports accessory", "sports equipment"],

    # --- winter sports ---
    "ski": ["ski", "winter sports", "sports equipment"],

    # --- playground ---
    "swing": ["swing", "playground", "sports equipment"],

    # ============================================================
    # L2 = furniture
    # ============================================================
    # --- seating ---
    "barber chair": ["chair", "seating", "furniture"],
    "folding chair": ["chair", "seating", "furniture"],
    "park bench": ["bench", "seating", "furniture"],
    "rocking chair": ["chair", "seating", "furniture"],
    "couch": ["sofa", "seating", "furniture"],
    "throne": ["chair", "seating", "furniture"],

    # --- bed ---
    "bassinet": ["cradle", "bed", "furniture"],
    "cradle": ["crib", "bed", "furniture"],
    "infant bed": ["crib", "bed", "furniture"],
    "four-poster bed": ["four-poster bed", "bed", "furniture"],

    # --- storage furniture ---
    "bookcase": ["shelf", "storage furniture", "furniture"],
    "chest": ["chest", "storage furniture", "furniture"],
    "chiffonier": ["cabinet", "storage furniture", "furniture"],
    "china cabinet": ["cabinet", "storage furniture", "furniture"],
    "filing cabinet": ["cabinet", "storage furniture", "furniture"],
    "medicine chest": ["cabinet", "storage furniture", "furniture"],
    "wardrobe": ["cabinet", "storage furniture", "furniture"],

    # --- table ---
    "desk": ["desk", "table", "furniture"],
    "dining table": ["dining table", "table", "furniture"],

    # --- cabinet ---
    "entertainment center": ["entertainment center", "cabinet", "furniture"],

    # --- home accessory ---
    "fire screen sheet": ["fireplace accessory", "home accessory", "furniture"],
    "window screen": ["screen", "home accessory", "furniture"],
    "window shade": ["shade", "home accessory", "furniture"],

    # --- decoration ---
    "jack-o'-lantern": ["pumpkin", "decoration", "furniture"],
    "maypole": ["pole", "decoration", "furniture"],
    "vase": ["vessel", "decoration", "furniture"],

    # --- lighting ---
    "lampshade": ["lighting accessory", "lighting", "furniture"],
    "spotlight": ["light", "lighting", "furniture"],
    "table lamp": ["lamp", "lighting", "furniture"],
    "torch": ["light", "lighting", "furniture"],

    # --- support ---
    "pedestal": ["pedestal", "support", "furniture"],

    # ============================================================
    # L2 = kitchenware
    # ============================================================
    # --- utensil ---
    "beer glass": ["glass", "utensil", "kitchenware"],
    "cauldron": ["pot", "utensil", "kitchenware"],
    "cleaver": ["knife", "utensil", "kitchenware"],
    "cocktail shaker": ["barware", "utensil", "kitchenware"],
    "coffee mug": ["mug", "utensil", "kitchenware"],
    "corkscrew": ["opener", "utensil", "kitchenware"],
    "Dutch oven": ["pot", "utensil", "kitchenware"],
    "frying pan": ["pan", "utensil", "kitchenware"],
    "goblet": ["drinkware", "utensil", "kitchenware"],
    "ladle": ["ladle", "utensil", "kitchenware"],
    "measuring cup": ["measuring tool", "utensil", "kitchenware"],
    "mixing bowl": ["bowl", "utensil", "kitchenware"],
    "mortar": ["grinding tool", "utensil", "kitchenware"],
    "plate rack": ["rack", "utensil", "kitchenware"],
    "salt shaker": ["shaker", "utensil", "kitchenware"],
    "soup bowl": ["bowl", "utensil", "kitchenware"],
    "spatula": ["spatula", "utensil", "kitchenware"],
    "strainer": ["sieve", "utensil", "kitchenware"],
    "teapot": ["pot", "utensil", "kitchenware"],
    "tray": ["tray", "utensil", "kitchenware"],
    "wok": ["pan", "utensil", "kitchenware"],
    "wooden spoon": ["spoon", "utensil", "kitchenware"],
    "plate": ["dishware", "utensil", "kitchenware"],
    "cup": ["cup", "utensil", "kitchenware"],

    # --- appliance ---
    "coffeemaker": ["coffeemaker", "appliance", "kitchenware"],
    "Crock Pot": ["slow cooker", "appliance", "kitchenware"],
    "dishwasher": ["dishwasher", "appliance", "kitchenware"],
    "espresso machine": ["espresso machine", "appliance", "kitchenware"],
    "microwave oven": ["oven", "appliance", "kitchenware"],
    "refrigerator": ["refrigerator", "appliance", "kitchenware"],
    "stove": ["stove", "appliance", "kitchenware"],
    "toaster": ["toaster", "appliance", "kitchenware"],
    "waffle iron": ["waffle iron", "appliance", "kitchenware"],

    # ============================================================
    # L2 = tool
    # ============================================================
    # --- measurement_instrument ---
    "abacus": ["calculating device", "measurement_instrument", "tool"],
    "analog clock": ["clock", "measurement_instrument", "tool"],
    "barometer": ["measuring instrument", "measurement_instrument", "tool"],
    "binoculars": ["optical instrument", "measurement_instrument", "tool"],
    "digital clock": ["clock", "measurement_instrument", "tool"],
    "digital watch": ["watch", "measurement_instrument", "tool"],
    "hourglass": ["timepiece", "measurement_instrument", "tool"],
    "loupe": ["magnifier", "measurement_instrument", "tool"],
    "magnetic compass": ["compass", "measurement_instrument", "tool"],
    "radio telescope": ["telescope", "measurement_instrument", "tool"],
    "ruler": ["measuring tool", "measurement_instrument", "tool"],
    "weighing scale": ["scale", "measurement_instrument", "tool"],
    "slide rule": ["calculating device", "measurement_instrument", "tool"],
    "stopwatch": ["timepiece", "measurement_instrument", "tool"],
    "sundial": ["timepiece", "measurement_instrument", "tool"],
    "wall clock": ["clock", "measurement_instrument", "tool"],

    # --- weapon ---
    "assault rifle": ["rifle", "weapon", "tool"],
    "cannon": ["artillery", "weapon", "tool"],
    "holster": ["weapon accessory", "weapon", "tool"],
    "missile": ["projectile", "weapon", "tool"],
    "projectile": ["weapon part", "weapon", "tool"],
    "revolver": ["handgun", "weapon", "tool"],
    "rifle": ["firearm", "weapon", "tool"],
    "scabbard": ["weapon accessory", "weapon", "tool"],
    "shield": ["armor", "weapon", "tool"],

    # --- stationery ---
    "ballpoint pen": ["pen", "stationery", "tool"],
    "ring binder": ["ring binder", "stationery", "tool"],
    "fountain pen": ["pen", "stationery", "tool"],
    "pencil case": ["case", "stationery", "tool"],
    "pencil sharpener": ["sharpener", "stationery", "tool"],
    "quill": ["pen", "stationery", "tool"],
    "eraser": ["eraser", "stationery", "tool"],
    "website": ["webpage", "stationery", "tool"],
    "comic book": ["book", "stationery", "tool"],
    "crossword": ["puzzle", "stationery", "tool"],
    "dust jacket": ["book cover", "stationery", "tool"],
    "menu": ["printed matter", "stationery", "tool"],

    # --- medical_device ---
    "Band-Aid": ["medical supply", "medical_device", "tool"],
    "crutch": ["mobility aid", "medical_device", "tool"],
    "neck brace": ["medical device", "medical_device", "tool"],
    "oxygen mask": ["mask", "medical_device", "tool"],
    "stethoscope": ["medical instrument", "medical_device", "tool"],
    "stretcher": ["medical equipment", "medical_device", "tool"],
    "syringe": ["medical instrument", "medical_device", "tool"],

    # --- hand_tool ---
    "wheelbarrow": ["cart", "hand_tool", "tool"],
    "broom": ["cleaning tool", "hand_tool", "tool"],
    "candle": ["light source", "hand_tool", "tool"],
    "can opener": ["kitchen tool", "hand_tool", "tool"],
    "hammer": ["hammer", "hand_tool", "tool"],
    "hatchet": ["axe", "hand_tool", "tool"],
    "knot": ["textile knot", "hand_tool", "tool"],
    "paper knife": ["knife", "hand_tool", "tool"],
    "lighter": ["ignition device", "hand_tool", "tool"],
    "match": ["ignition device", "hand_tool", "tool"],
    "mousetrap": ["trap", "hand_tool", "tool"],
    "muzzle": ["animal accessory", "hand_tool", "tool"],
    "paintbrush": ["brush", "hand_tool", "tool"],
    "parachute": ["safety equipment", "hand_tool", "tool"],
    "hand plane": ["plane", "hand_tool", "tool"],
    "plow": ["agricultural tool", "hand_tool", "tool"],
    "plunger": ["bathroom tool", "hand_tool", "tool"],
    "potter's wheel": ["pottery equipment", "hand_tool", "tool"],
    "reel": ["spool", "hand_tool", "tool"],
    "screwdriver": ["hand tool", "hand_tool", "tool"],
    "shovel": ["digging tool", "hand_tool", "tool"],
    "spindle": ["textile tool", "hand_tool", "tool"],
    "mop": ["cleaning tool", "hand_tool", "tool"],
    "thimble": ["sewing tool", "hand_tool", "tool"],
    "tripod": ["stand", "hand_tool", "tool"],
    "bubble": ["fluid structure", "hand_tool", "tool"],

    # --- laboratory_equipment ---
    "beaker": ["glassware", "laboratory_equipment", "tool"],
    "Petri dish": ["laboratory equipment", "laboratory_equipment", "tool"],

    # --- hardware ---
    "bottle cap": ["closure", "hardware", "tool"],
    "buckle": ["fastener", "hardware", "tool"],
    "cassette": ["media storage", "hardware", "tool"],
    "chain": ["chain", "hardware", "tool"],
    "coil": ["coil", "hardware", "tool"],
    "grille": ["car part", "hardware", "tool"],
    "hook": ["hook", "hardware", "tool"],
    "lens cap": ["camera accessory", "hardware", "tool"],
    "nail": ["fastener", "hardware", "tool"],
    "safety pin": ["pin", "hardware", "tool"],
    "screw": ["fastener", "hardware", "tool"],
    "solar thermal collector": ["solar equipment", "hardware", "tool"],
    "switch": ["electrical component", "hardware", "tool"],

    # --- power_tool ---
    "automated teller machine": ["financial machine", "power_tool", "tool"],
    "chainsaw": ["saw", "power_tool", "tool"],
    "crane (machine)": ["construction machine", "power_tool", "tool"],
    "gas pump": ["fuel equipment", "power_tool", "tool"],
    "guillotine": ["execution device", "power_tool", "tool"],
    "harvester": ["agricultural machine", "power_tool", "tool"],
    "lawn mower": ["garden machine", "power_tool", "tool"],
    "power drill": ["drill", "power_tool", "tool"],
    "slot machine": ["gambling machine", "power_tool", "tool"],
    "threshing machine": ["agricultural machine", "power_tool", "tool"],
    "vending machine": ["dispensing machine", "power_tool", "tool"],

    # --- appliance ---
    "electric fan": ["fan", "appliance", "tool"],
    "clothes iron": ["clothes iron", "appliance", "tool"],
    "radiator": ["heating device", "appliance", "tool"],
    "rotisserie": ["cooking equipment", "appliance", "tool"],
    "sewing machine": ["sewing machine", "appliance", "tool"],
    "space heater": ["heater", "appliance", "tool"],
    "vacuum cleaner": ["vacuum cleaner", "appliance", "tool"],
    "washing machine": ["washing machine", "appliance", "tool"],

    # --- personal_care ---
    "face powder": ["cosmetic", "personal_care", "tool"],
    "hair spray": ["cosmetic", "personal_care", "tool"],
    "hair dryer": ["appliance", "personal_care", "tool"],
    "lipstick": ["cosmetic", "personal_care", "tool"],
    "lotion": ["cosmetic", "personal_care", "tool"],
    "nipple": ["baby accessory", "personal_care", "tool"],
    "paper towel": ["paper towel", "personal_care", "tool"],
    "perfume": ["fragrance", "personal_care", "tool"],
    "soap dispenser": ["dispenser", "personal_care", "tool"],
    "sunscreen": ["cosmetic", "personal_care", "tool"],
    "toilet paper": ["paper product", "personal_care", "tool"],
}
