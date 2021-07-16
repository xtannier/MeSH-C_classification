# coding=utf8
import os
import argparse
import pandas
import pickle
import stanza
import sklearn
from sklearn.ensemble import AdaBoostClassifier

# Les parametres extraits du corpus d'entrainement sont stockes dans 'data/work/patient_sex_features.csv' et 'data/work/patient_age_features.csv'
# Les annotations sont dans 'data/work/patient_classes.csv'

# Facteur d'attenuation du poids des indices selon leur position, entre le debut et la fin des textes
att = 2

# Liste de prenoms genre's
prenoms = { 'Abdallah': "M", 'Abdel': "M", 'Abdelaziz': "M", 'Abdelhafid': "M", 'Abdelhak': "M", 'Abdelhamid': "M",
            'Abdelkader': "M", 'Abdelkrim': "M", 'Abdellah': "M", 'Abdellatif': "M", 'Abdelmajid': "M", 'Abderrahim': "M",
            'Abderrahman': "M", 'Abderrahmane': "M", 'Abderrazak': "M", 'Abdeslam': "M", 'Abdoulaye': "M", 'Abel': "M",
            'Abilio': "M", 'Abraham': "M", 'Achille': "M", 'Achour': "M", 'Adam': "M", 'Adel': "M",
            'Adèle': "F", 'Adeline': "F", 'Adolphe': "M", 'Adriano': "M", 'Adrien': "M", 'Adrienne': "F",
            'Agathe': "F", 'Agnès': "F", 'Agostinho': "M", 'Agostino': "M", 'Ahcène': "M", 'Ahmad': "M",
            'Ahmed': "M", 'Ahmet': "M", 'Aicha': "F", 'Aimable': "M", 'Aimé': "M", 'Aimée': "F",
            'Aissa': "F", 'Akli': "M", 'Alain': "M", 'Alain-Michel': "M", 'Alain-Pierre': "M", 'Alan': "M",
            'Alban': "M", 'Albano': "M", 'Albert': "M", 'Alberte': "F", 'Albertine': "F", 'Alberto': "M",
            'Albin': "M", 'Albino': "M", 'Alcide': "M", 'Aldo': "M", 'Alex': "M", 'Alexandra': "F",
            'Alexandre': "M", 'Alexandrine': "F", 'Alexia': "F", 'Alexis': "M", 'Alfonso': "M", 'Alfred': "M",
            'Alfréda': "F", 'Alfredo': "M", 'Ali': "M", 'Alice': "F", 'Alicia': "F", 'Aliette': "F",
            'Aline': "F", 'Alix': "F", 'Allal': "M", 'Aloyse': "M", 'Alphonse': "M", 'Alphonsine': "F",
            'Alvaro': "M", 'Amadou': "M", 'Amand': "M", 'Amar': "M", 'Amara': "F", 'Amaury': "M",
            'Ambre': "F", 'Ambroise': "M", 'Amédé': "M", 'Amédée': "M", 'Amélia': "F", 'Amélie': "F",
            'Américo': "M", 'Amina': "F", 'Ammar': "M", 'Amor': "M", 'Ana': "F", 'Anaïs': "F",
            'Anatole': "M", 'Andrea': "F", 'Andréa': "F", 'Andree': "F", 'Andrée': "F", 'André-Jean': "M",
            'André-Marie': "M", 'Andrew': "M", 'Ange': "M", 'Angel': "M", 'Angéla': "F", 'Angèle': "F",
            'Angélina': "F", 'Angéline': "F", 'Angélique': "F", 'Angélo': "M", 'Ange-Marie': "M", 'Anibal': "M",
            'Anicet': "M", 'Anita': "F", 'Anna': "F", 'Anna-Bella': "F", 'Annabelle': "F", 'Anna-Belle': "F",
            'Anna-Elisa': "F", 'Anna-Maria': "F", 'Anne': "F", 'Anne-Aymone': "F", 'Anne-Catherine': "F", 'Anne-Cécile': "F",
            'Anne-Charlotte': "F", 'Anne-Christèle': "F", 'Anne-Claire': "F", 'Anne-Elisabeth': "F", 'Anne-Isabelle': "F", 'Anne-Juliette': "F",
            'Anne-Laure': "F", 'Anne-Lise': "F", 'Anne-Marceline': "F", 'Anne-Marguerite': "F", 'Anne-Marie': "F", 'Anne-Pascale': "F",
            'Anne-Pénélope': "F", 'Anne-Sophie': "F", 'Annette': "F", 'Anne-Yvonne': "F", 'Annick': "F", 'Annie': "F",
            'Annie-Christine': "F", 'Annie-Claire': "F", 'Annie-Claude': "F", 'Annie-Flore': "F", 'Annie-France': "F", 'Annie-Françoise': "F",
            'Annie-Laure': "F", 'Annie-Paule': "F", 'Annie-Pierre': "F", 'Annie-Rose': "F", 'Annie-Thérèse': "F", 'Anny': "F",
            'Anselme': "M", 'Anthea': "F", 'Anthéa': "F", 'Anthony': "M", 'Antoine': "M", 'Antoinette': "F",
            'Antonia': "F", 'Antonin': "M", 'Antonine': "F", 'Antonino': "M", 'Antonio': "M", 'Antony': "M",
            'Ariane': "F", 'Ariel': "M", 'Arielle': "F", 'Aristide': "M", 'Arlette': "F", 'Armand': "M",
            'Armande': "F", 'Armandine': "F", 'Armando': "M", 'Armel': "M", 'Armelle': "F", 'Armindo': "M",
            'Arnaldo': "M", 'Arnaud': "M", 'Arnold': "M", 'Arsène': "M", 'Arthur': "M", 'Astrid': "F",
            'Attilio': "M", 'Aude': "F", 'Audrey': "F", 'Audrey-Anne': "F", 'Augusta': "F", 'Auguste': "M",
            'Augustin': "M", 'Augustine': "F", 'Augusto': "M", 'Aurélie': "F", 'Aurélien': "M", 'Aurélio': "M",
            'Aurore': "F", 'Avelino': "M", 'Axel': "F", 'Aziz': "M", 'Aziza': "F", 'Bachir': "M",
            'Baptiste': "M", 'Baptistine': "F", 'Barbara': "F", 'Basile': "M", 'Baya': "F", 'Béatrice': "F",
            'Béatrix': "F", 'Beatrix-Laure': "F", 'Bechir': "M", 'Belgacem': "M", 'Belkacem': "M", 'Belle-Alice': "F",
            'Belle-Helene': "F", 'Bénédicte': "F", 'Bénito': "M", 'Benjamin': "M", 'Benoist': "M", 'Benoît': "M",
            'Bérangère': "F", 'Bérénice-Marie': "F", 'Bernadette': "F", 'Bernard': "M", 'Berthe': "F", 'Bertin': "M",
            'Bertrand': "M", 'Bettina': "F", 'Betty': "F", 'Bien-Aimé': "M", 'Blaise': "M", 'Blanche': "F",
            'Blandine': "F", 'Boris': "M", 'Boualem': "M", 'Bouzid': "M", 'Brahim': "M", 'Brian': "M",
            'Brice': "M", 'Brigitte': "F", 'Bruna': "F", 'Bruno': "M", 'Calixte': "F", 'Calogero': "M",
            'Camille': "?", 'Candido': "M", 'Carine': "F", 'Carl': "M", 'Carla': "F", 'Carlo': "M",
            'Carlos': "M", 'Carmel': "M", 'Carmela': "F", 'Carmelo': "M", 'Carmen': "F", 'Carmine': "F",
            'Carol': "M", 'Carole': "F", 'Carole-Anne': "F", 'Caroline': "F", 'Casimir': "M", 'Cassandra': "F",
            'Cassandre': "F", 'Cataldo': "M", 'Catherine': "F", 'Catherine-Amelie': "F", 'Catherine-Josée': "F", 'Catherine-Lune': "F",
            'Cathy': "F", 'Cécile': "F", 'Celia': "F", 'Célia': "F", 'Céline': "F", 'Chantal': "F",
            'Charles': "M", 'Charles-Étienne': "M", 'Charles-François-Bienven': "M", 'Charles-Henri': "M", 'Charles-Olivier': "M", 'Charlette': "F",
            'Charlie': "M", 'Charline': "F", 'Charlotte': "F", 'Charly': "M", 'Cherif': "M", 'Christel': "F",
            'Christèle': "F", 'Christelle': "F", 'Christian': "M", 'Christiane': "F", 'Christian-Jacques': "M", 'Christianne': "F",
            'Christina': "F", 'Christine': "F", 'Christophe': "M", 'Christopher': "M", 'Chrystel': "F", 'Chrystèle': "F",
            'Chrystelle': "F", 'Claire': "F", 'Claire-Juliette': "F", 'Clara': "F", 'Clarisse': "F", 'Claude': "?",
            'Claude-Henri': "M", 'Claude-Joël': "M", 'Claude-Marie': "M", 'Claude-Pierre': "M", 'Claudette': "F", 'Claudia': "F",
            'Claudie': "F", 'Claudine': "F", 'Claudio': "M", 'Claudius': "M", 'Claudy': "F", 'Clémence': "F",
            'Clément': "M", 'Clémentine': "F", 'Clotaire': "M", 'Clothilde': "F", 'Clotilde': "F", 'Clovis': "M",
            'Colette': "F", 'Colin': "M", 'Colombe': "F", 'Conception': "F", 'Constance': "F", 'Constant': "M",
            'Constantin': "M", 'Coralie': "F", 'Corentin': "M", 'Corine': "F", 'Corinne': "F", 'Cosimo': "M",
            'Cyprien': "M", 'Cyril': "M", 'Cyrille': "?", 'Daisy': "F", 'Dalila': "F", 'Damien': "M",
            'Danièl': "M", 'Danièle': "F", 'Danièlle': "F", 'Dante': "M", 'Dany': "?", 'Dario': "M",
            'David': "M", 'David-Frédéric': "M", 'David-Vincent': "M", 'Delphin': "M", 'Delphine': "F", 'Denis': "M",
            'Denise': "F", 'Denys': "M", 'Denyse': "F", 'Diamantino': "M", 'Diana': "F", 'Diane': "F",
            'Didier': "M", 'Diégo': "M", 'Dimitri': "M", 'Dina': "F", 'Dino': "M", 'Djamel': "M",
            'Djamila': "F", 'Djelloul': "M", 'Djilali': "M", 'Dolorès': "F", 'Domenico': "M", 'Domingo': "M",
            'Domingos': "M", 'Dominique': "?", 'Donald': "M", 'Donat': "M", 'Donatien': "M", 'Donato': "M",
            'Dora': "F", 'Dorian': "M", 'Doris': "F", 'Dorothée': "F", 'Driss': "M", 'Dylan': "M",
            'Eddy': "M", 'Edgar': "M", 'Edgard': "M", 'Edith': "F", 'Edmée': "?", 'Edmond': "M",
            'Edmonde': "F", 'Edouard': "M", 'Edouardo': "M", 'Eduardo': "M", 'Edward': "M", 'Edwige': "F",
            'Egidio': "M", 'Eglantine': "F", 'Eléonore': "F", 'Elia': "F", 'Elian': "M", 'Eliane': "F",
            'Elias': "M", 'Elie': "M", 'Eliette': "F", 'Elio': "M", 'Elisa': "F", 'Elisabeth': "F",
            'Élisa-Maude': "F", 'Elise': "F", 'Elisée': "M", 'Elizabeth': "F", 'Emile': "M", 'Emilia': "F",
            'Emilie': "F", 'Emilien': "M", 'Emilienne': "F", 'Emilio': "M", 'Emma': "F", 'Emma-Line': "F",
            'Emma-Lou': "F", 'Emma-Louise': "F", 'Emmanuel': "M", 'Emmanuelle': "F", 'Emma-Rose': "F", 'Enrico': "M",
            'Enrique': "M", 'Enzo': "M", 'Eric': "M", 'Erick': "M", 'Erik': "M", 'Erika': "F",
            'Erna': "F", 'Ernest': "M", 'Ernestine': "F", 'Ernesto': "M", 'Erwan': "M", 'Erwin': "M",
            'Espérance': "F", 'Estelle': "F", 'Esther': "F", 'Etienne': "M", 'Etiennette': "F", 'Eugène': "M",
            'Eugénie': "F", 'Eulalie': "F", 'Eva': "F", 'Evariste': "M", 'Èva-Rose': "F", 'Eve': "F",
            'Eve-Charlotte': "F", 'Eveline': "F", 'Evelyne': "F", 'Ève-Marie': "F", 'Fabien': "M", 'Fabienne': "F",
            'Fabrice': "M", 'Fadila': "F", 'Fanny': "F", 'Farid': "M", 'Farida': "F", 'Fathi': "M",
            'Fathia': "F", 'Fatiha': "F", 'Fatima': "F", 'Fatma': "F", 'Fatna': "F", 'Fausto': "M",
            'Félicie': "F", 'Félicien': "M", 'Félicité': "F", 'Félix': "M", 'Ferdinand': "M", 'Fernand': "M",
            'Fernanda': "F", 'Fernande': "F", 'Fernando': "M", 'Firmin': "M", 'Flavian': "M", 'Flavie': "F",
            'Flavien': "M", 'Fleury': "M", 'Flora': "F", 'Flore': "F", 'Floréal': "M", 'Florence': "F",
            'Florent': "M", 'Florentin': "M", 'Florentine': "F", 'Florian': "M", 'Floriane': "F", 'Fortuné': "M",
            'Fortunée': "F", 'Fouad': "M", 'Fouzia': "F", 'Franc': "M", 'France': "F", 'Francesca': "F",
            'Francesco': "M", 'Francette': "F", 'Francine': "F", 'Francis': "M", 'Francisca': "F", 'Francisco': "M",
            'Francisque': "M", 'Francis-Xavier': "M", 'Franck': "M", 'Franck-Emmanuel': "M", 'Franck-Olivier': "M", 'Franco': "M",
            'François': "M", 'François-Alexandre': "M", 'François-Charles': "M", 'François-David': "M", 'Françoise': "F", 'François-Éric': "M",
            'François-Guillaume': "M", 'François-Henri': "M", 'François-Jérôme': "M", 'François-Joseph': "M", 'François-Julien': "M", 'François-Louis': "M",
            'François-Marie': "M", 'François-Nicolas': "M", 'François-Pierre': "M", 'François-Régis': "M", 'François-René': "M", 'François-Xavier': "M",
            'Frank': "M", 'Frank-Erwann': "M", 'Frank-Louis': "M", 'Frank-Nicolas': "M", 'Frantz': "M", 'Franz': "M",
            'Fred': "M", 'Freddy': "M", 'Frederic': "M", 'Frédéric-François': "M", 'Frederique': "F", 'Fredy': "M",
            'Frieda': "F", 'Gabriel': "M", 'Gabrielle': "F", 'Gaby': "F", 'Gaël': "M", 'Gaëlle': "F",
            'Gaëtan': "M", 'Gaspard': "M", 'Gaston': "M", 'Geneviève': "F", 'Geoffroy': "M", 'George': "M",
            'Georges': "M", 'Georget': "M", 'Georgette': "F", 'Georgina': "F", 'Gérald': "M", 'Géraldine': "F",
            'Gérard': "M", 'Germain': "M", 'Germaine': "F", 'Gertrude': "F", 'Gervais': "M", 'Géry': "M",
            'Ghislain': "M", 'Ghislaine': "F", 'Ghyslaine': "F", 'Gil': "M", 'Gilbert': "M", 'Gilberte': "F",
            'Gilda': "F", 'Gilette': "F", 'Gilles': "M", 'Gillette': "F", 'Gina': "F", 'Ginette': "F",
            'Gino': "M", 'Giorgio': "M", 'Giovanna': "F", 'Giovanni': "M", 'Gisèle': "F", 'Giselle': "F",
            'Gislaine': "F", 'Giuseppe': "M", 'Gladys': "F", 'Gloria': "F", 'Gonzague': "M", 'Gracieuse': "F",
            'Gratien': "M", 'Graziella': "F", 'Grégoire': "M", 'Grégory': "M", 'Guido': "M", 'Guilaine': "F",
            'Guillaume': "M", 'Guillemette': "F", 'Gunther': "M", 'Gustave': "M", 'Guy': "M", 'Guylaine': "F",
            'Guylène': "F", 'Guy-Marie': "M", 'Guy-Noël': "M", 'Habib': "M", 'Habiba': "F", 'Hachemi': "M",
            'Hadda': "F", 'Hadj': "M", 'Hafida': "F", 'Halil': "M", 'Halima': "F", 'Hamadi': "M",
            'Hamed': "M", 'Hamid': "M", 'Hannah-Belle': "F", 'Hans': "M", 'Harold': "M", 'Harry': "M",
            'Hasan': "M", 'Hassan': "M", 'Hassen': "M", 'Hector': "M", 'Hedi': "M", 'Heinz': "M",
            'Héléna': "F", 'Hélène': "F", 'Helene-Sarah': "F", 'Helga': "F", 'Heloise-Luce': "F", 'Henri': "M",
            'Henri-Claude': "M", 'Henriette': "F", 'Henri-Jean': "M", 'Henri-Louis': "M", 'Henri-Michel': "M", 'Henri-Pierre': "M",
            'Henrique': "M", 'Henri-Xavier': "M", 'Henry': "M", 'Herbert': "M", 'Hermann': "M", 'Hermine': "F",
            'Hervé': "M", 'Hilaire': "M", 'Hilda': "F", 'Hippolyte': "M", 'Hocine': "M", 'Honoré': "M",
            'Honorine': "F", 'Horace': "M", 'Horacio': "M", 'Horst': "M", 'Hortense': "F", 'Houcine': "M",
            'Houria': "F", 'Hubert': "M", 'Hugo': "M", 'Hugues': "M", 'Huguette': "F", 'Humbert': "M",
            'Hyacinthe': "F", 'Ian': "M", 'Ibrahim': "M", 'Ibrahima': "F", 'Ida': "F", 'Idir': "M",
            'Ignace': "M", 'Igor': "M", 'Ilda': "F", 'Inès': "F", 'Ingrid': "F", 'Irène': "F",
            'Irénée': "M", 'Irma': "F", 'Isaac': "M", 'Isabel': "F", 'Isabelle': "F", 'Isidore': "M",
            'Ismaël': "M", 'Ismail': "M", 'Italo': "M", 'Ivan': "M", 'Jack': "M", 'Jacki': "?",
            'Jackie': "?", 'Jacky': "?", 'Jacob': "M", 'Jacqueline': "F", 'Jacques': "M", 'Jacques-Alexandre': "M",
            'Jacques-Antoine': "M", 'Jacques-Édouard': "M", 'Jacques-Henri': "M", 'Jacques-Olivier': "M", 'Jacques-Pierre': "M", 'Jacques-Yves': "M",
            'Jacquy': "?", 'Jad': "M", 'Jade': "F", 'Jamal': "M", 'Jamila': "F", 'Jane': "F",
            'Janick': "?", 'Janine': "F", 'Jannick': "F", 'Jany': "F", 'Jean': "M", 'Jean-Adrien': "M",
            'Jean-Aimé': "M", 'Jean-Alain': "M", 'Jean-Albert': "M", 'Jean-Alexandre': "M", 'Jean-Alexis': "M", 'Jean-Alfred': "M",
            'Jean-André': "M", 'Jean-Anthony': "M", 'Jean-Antoine': "M", 'Jean-Arnaud': "M", 'Jean-Auguste': "M", 'Jean-Baptiste': "M",
            'Jean-Barthélémy': "M", 'Jean-Bastien': "M", 'Jean-Benoît': "M", 'Jean-Bernard': "M", 'Jean-Bertrand': "M", 'Jean-Blaise': "M",
            'Jean-Bosco': "M", 'Jean-Briac': "M", 'Jean-Brice': "M", 'Jean-Bruno': "M", 'Jean-Camille': "M", 'Jean-Carlo': "M",
            'Jean-Carlos': "M", 'Jean-Casimir': "M", 'Jean-Cédric': "M", 'Jean-Charles': "M", 'Jean-Christian': "M", 'Jean-Christophe': "M",
            'Jean-Clair': "M", 'Jean-Claude': "M", 'Jean-Clément': "M", 'Jean-Côme': "M", 'Jean-Cyril': "M", 'Jean-Cyrille': "M",
            'Jean-Damien': "M", 'Jean-Daniel': "M", 'Jean-David': "M", 'Jean-Denis': "M", 'Jean-Didier': "M", 'Jean-Dominique': "M",
            'Jean-Édouard': "M", 'Jean-Élie': "M", 'Jean-Éloi': "M", 'Jean-Émile': "M", 'Jean-Éric': "M", 'Jean-Étienne': "M",
            'Jean-Eudes': "M", 'Jean-Francis': "M", 'Jean-François': "M", 'Jean-Gabriel': "M", 'Jean-Guy': "M", 'Jeanine': "F",
            'Jean-Jacques': "M", 'Jean-Julien': "M", 'Jean-Kévin': "M", 'Jean-Louis': "M", 'Jean-Loup': "M", 'Jean-Luc': "M",
            'Jean-Marc': "M", 'Jean-Marcel': "M", 'Jean-Marie': "M", 'Jean-Mathieu': "M", 'Jean-Maurice': "M", 'Jean-Maxime': "M",
            'Jean-Michel': "M", 'Jeanne': "F", 'Jeanne-Antide': "F", 'Jeanne-Charlotte': "F", 'Jeanne-Claire': "F", 'Jeanne-Françoise': "F",
            'Jeanne-Hélène': "F", 'Jeanne-Laure': "F", 'Jeanne-Lise': "F", 'Jeanne-Louise': "F", 'Jeanne-Marie': "F", 'Jeanne-Sophie': "F",
            'Jeannette': "F", 'Jeannick': "?", 'Jean-Nicolas': "M", 'Jeannie': "F", 'Jeannine': "F", 'Jean-Noël': "M",
            'Jeannot': "M", 'Jean-Pascal': "M", 'Jean-Patrick': "M", 'Jean-Paul': "M", 'Jean-Philibert': "M", 'Jean-Philippe': "M",
            'Jean-Pierre': "M", 'Jean-Raymond': "M", 'Jean-René': "M", 'Jean-Robert': "M", 'Jean-Roch': "M", 'Jean-Sébastien': "M",
            'Jean-Simon': "M", 'Jean-Thomas': "M", 'Jean-Vincent': "M", 'Jean-Xavier': "M", 'Jean-Yves': "M", 'Jennifer': "F",
            'Jenny': "F", 'Jérôme': "M", 'Jérôme-André': "M", 'Jim': "M", 'Jimmy': "M", 'Jo': "?",
            'Joachim': "M", 'Joannes': "M", 'Joanny': "?", 'João': "M", 'Joaquim': "M", 'Jocelyn': "M",
            'Jocelyne': "F", 'Joël': "M", 'Joëlle': "F", 'Johan': "M", 'Johann': "M", 'Johanna': "F",
            'John': "M", 'Johnny': "M", 'Jonathan': "M", 'Jordan': "M", 'Jorge': "M", 'José': "?",
            'Josée': "F", 'Josée-Anne': "F", 'Joseph': "M", 'Josépha': "F", 'Joséphine': "F", 'Josette': "F",
            'Josiane': "F", 'Josseline': "F", 'Josy': "F", 'Josyane': "F", 'Juan': "M", 'Juana': "F",
            'Judith': "F", 'Jules': "M", 'Julia': "F", 'Julian': "M", 'Julie': "F", 'Julie-Maude': "F",
            'Julie-Michelle': "F", 'Julien': "M", 'Julienne': "F", 'Julien-Pierre': "M", 'Juliette': "F", 'Julio': "M",
            'Justin': "M", 'Justine': "F", 'Kaddour': "M", 'Kader': "M", 'Kamal': "M", 'Kamel': "M",
            'Karen': "F", 'Karim': "M", 'Karima': "F", 'Karin': "F", 'Karine': "F", 'Karl': "M",
            'Katherine': "F", 'Kathy': "F", 'Katia': "F", 'Katy': "F", 'Kenneth': "M", 'Kenza': "F",
            'Kevin': "M", 'Khadija': "F", 'Khaled': "M", 'Khalid': "M", 'Khedidja': "F", 'Killian': "M",
            'Kim': "M", 'Kleber': "M", 'Kouider': "M", 'Ladislas': "M", 'Laëtitia': "F", 'Lahcen': "M",
            'Lahoucine': "M", 'Lakdar': "M", 'Lakhdar': "M", 'Lambert': "M", 'Larbi': "M", 'Latifa': "F",
            'Laura': "F", 'Laure': "F", 'Laure-Élise': "F", 'Laure-Lise': "F", 'Laure-Lou': "F", 'Laurence': "F",
            'Laurent': "M", 'Laurette': "F", 'Laurie': "F", 'Laurie-Anne': "F", 'Laurine': "F", 'Layla': "F",
            'Lazare': "M", 'Léa': "F", 'Léa-Jade': "F", 'Léandre': "M", 'Leila': "F", 'Leïla': "F",
            'Leyla': "F", 'Lena': "F", 'Léna': "F", 'Lena-Lys': "F", 'Léo': "M", 'Léocadie': "F",
            'Léon': "M", 'Léonard': "M", 'Léonardo': "M", 'Léonce': "M", 'Léone': "F", 'Léonie': "F",
            'Léonne': "F", 'Léontine': "F", 'Léo-Paul': "M", 'Léopold': "M", 'Lidia': "F", 'Lila': "F",
            'Lilian': "M", 'Liliane': "F", 'Lilou': "F", 'Lily': "F", 'Lily-Rose': "F", 'Lina': "F",
            'Linda': "F", 'Line': "F", 'Lino': "M", 'Lionel': "M", 'Lionnel': "M", 'Lisa': "F",
            'Lisa-Marie': "F", 'Lise': "F", 'Lise-Anne': "F", 'Lisette': "F", 'Livio': "M", 'Loïc': "M",
            'Lola': "F", 'Lou-Anne': "F", 'Lou-Eve': "F", 'Louis': "M", 'Louisa': "F", 'Louis-Adolphe': "M",
            'Louis-André': "M", 'Louis-Antoine': "M", 'Louis-Arthur': "M", 'Louis-Benjamin': "M", 'Louis-Benoît': "M", 'Louis-Bernard': "M",
            'Louis-Camille': "M", 'Louis-Casimir': "M", 'Louis-Charles': "M", 'Louis-Daniel': "M", 'Louise': "F", 'Louise-Anne': "F",
            'Louise-Hélène': "F", 'Louise-Marie': "F", 'Louisette': "F", 'Louis-Franck': "M", 'Louis-Marie': "M", 'Louis-Philippe': "M",
            'Louis-Roger': "M", 'Louis-Valentin': "M", 'Louis-Xavier': "M", 'Luc': "M", 'Lucas': "M", 'Luce': "F",
            'Lucette': "F", 'Lucia': "F", 'Luciano': "M", 'Lucie': "F", 'Lucien': "M", 'Lucienne': "F",
            'Lucile': "F", 'Ludovic': "M", 'Luis': "M", 'Luisa': "F", 'Lydia': "F", 'Lydie': "F",
            'Lydie-Anne': "F", 'Lyliane': "F", 'Lynda': "F", 'Lysiane': "F", 'Mabrouk': "M", 'Madeleine': "F",
            'Maelle': "F", 'Maëlle': "F", 'Maeva': "F", 'Maëva': "F", 'Madjid': "M", 'Magali': "F",
            'Magalie': "F", 'Magdeleine': "F", 'Maguy': "F", 'Mahmoud': "M", 'Maïté': "F", 'Malik': "M",
            'Malika': "F", 'Mamadou': "M", 'Manfred': "M", 'Manon': "F", 'Mansour': "M", 'Manuel': "M",
            'Manuela': "F", 'Manuella': "F", 'Marc': "M", 'Marc-André': "M", 'Marc-Antoine': "M", 'Marceau': "M",
            'Marcel': "M", 'Marcelin': "M", 'Marceline': "F", 'Marcelle': "F", 'Marcellin': "M", 'Marcelline': "F",
            'Marco': "M", 'Margaret': "F", 'Margaux': "F", 'Margot': "F", 'Marguerite': "F", 'Margueritte': "F",
            'Maria': "F", 'Marian': "F", 'Mariane': "F", 'Marianne': "F", 'Mariano': "M", 'Marie': "F",
            'Marie-Alice': "F", 'Marie-Amélie': "F", 'Marie-Andrée': "F", 'Marie-Ange': "F", 'Marie-Anne': "F", 'Marie-Bérénice': "F",
            'Marie-Carmèle': "F", 'Marie-Chantal': "F", 'Marie-Charlotte': "F", 'Marie-Christine': "F", 'Marie-Claude': "F", 'Marie-Dominique': "F",
            'Marie-Eve': "F", 'Marie-France': "F", 'Marie-Françoise': "F", 'Marie-Hélène': "F", 'Marie-Isabelle': "F", 'Marie-Jeanne': "F",
            'Marie-Josée': "F", 'Marie-Joseph': "M", 'Marie-Josèphe': "F", 'Marie-Juliette': "F", 'Marie-Laure': "F", 'Marielle': "F",
            'Marie-Lou': "F", 'Marie-Louise': "F", 'Marie-Lydie': "F", 'Marie-Madeleine': "F", 'Marie-Pascale': "F", 'Marie-Paule': "F",
            'Marie-Pierre': "F", 'Marie-Raymonde': "F", 'Marie-Sarah': "F", 'Marie-Sophie': "F", 'Marie-Thérèse': "F", 'Mariette': "F",
            'Marie-Valentine': "F", 'Marie-Victoire': "F", 'Marie-Yolande': "F", 'Marilyne': "F", 'Marin': "M", 'Marina': "F",
            'Marine': "F", 'Marinette': "F", 'Marino': "M", 'Mario': "M", 'Marion': "F", 'Marius': "M",
            'Marjorie': "F", 'Mark': "M", 'Marlène': "F", 'Marlyse': "F", 'Martha': "F", 'Marthe': "F",
            'Martial': "M", 'Martin': "M", 'Martine': "F", 'Mary': "F", 'Marylène': "F", 'Maryline': "F",
            'Marylise': "F", 'Maryse': "F", 'Maryvonne': "F", 'Mathias': "M", 'Mathieu': "M", 'Mathilde': "F",
            'Mathis': "M", 'Mathurin': "M", 'Matthieu': "M", 'Maud': "F", 'Maurice': "M", 'Maurice-François': "M",
            'Mauricette': "F", 'Max': "M", 'Maxence': "M", 'Maxime': "M", 'Maximilien': "M", 'Maximin': "M",
            'Mehdi': "M", 'Mehmet': "M", 'Melissa': "F", 'Mélissa': "F", 'Mercédès': "F", 'Messaoud': "M",
            'Messaouda': "F", 'Meyer': "M", 'Mhamed': "M", 'Michaël': "M", 'Michel': "M", 'Michèle': "F",
            'Michelle': "F", 'Miloud': "M", 'Mimoun': "M", 'Mina': "F", 'Mireille': "F", 'Modeste': "M",
            'Mohamed': "M", 'Mohammad': "M", 'Mohammed': "M", 'Mohand': "M", 'Mohsen': "M", 'Moïse': "M",
            'Mokhtar': "M", 'Moktar': "M", 'Monica': "F", 'Monika': "F", 'Monique': "F", 'Morgane': "F",
            'Mostafa': "M", 'Mostefa': "M", 'Mouloud': "M", 'Mounir': "M", 'Mourad': "M", 'Moussa': "M",
            'Muguette': "F", 'Muriel': "F", 'Murielle': "F", 'Mustafa': "M", 'Mustapha': "M", 'Mylène': "F",
            'Myriam': "F", 'Nabil': "M", 'Nacer': "M", 'Nadège': "F", 'Nadia': "F", 'Nadine': "F",
            'Naïma': "F", 'Nancy': "F", 'Narcisse': "M", 'Nasser': "M", 'Natacha': "F", 'Nathalie': "F",
            'Nathalie-Anne': "F", 'Nello': "M", 'Nelly': "F", 'Nelson': "M", 'Nestor': "M", 'Nicolas': "M",
            'Nicole': "F", 'Nicolle': "F", 'Nina': "F", 'Noël': "M", 'Noëlie': "F", 'Noëlla': "F",
            'Noëlle': "F", 'Noémie': "F", 'Nora': "F", 'Norbert': "M", 'Nordine': "M", 'Noureddine': "M",
            'Nourredine': "M", 'Océane': "F", 'Octave': "M", 'Octavie': "F", 'Odette': "F", 'Odile': "F",
            'Olga': "F", 'Olivia': "F", 'Olivier': "M", 'Olympe': "F", 'Omar': "M", 'Omer': "M",
            'Oreste': "M", 'Orlando': "M", 'Oscar': "M", 'Osman': "M", 'Oswald': "M", 'Otto': "M",
            'Oumar': "M", 'Pablo': "M", 'Palmyre': "F", 'Paola': "F", 'Paolo': "M", 'Pascal': "M",
            'Pascale': "F", 'Pascaline': "F", 'Patrice': "M", 'Patricia': "F", 'Patrick': "M", 'Paul': "M",
            'Paula': "F", 'Paul-André': "M", 'Paul-Antoine': "M", 'Paul-Armand': "M", 'Paul-Arthur': "M", 'Paule': "F",
            'Paule-Émeline': "F", 'Paule-Marie': "F", 'Paulette': "F", 'Paul-Henri': "M", 'Paulin': "M", 'Pauline': "F",
            'Paul-Marie': "M", 'Paulo': "M", 'Paul-Vincent': "M", 'Pédro': "M", 'Peggy': "F", 'Pénélope-Fiona': "F",
            'Perrine': "F", 'Peter': "M", 'Philibert': "M", 'Philip': "M", 'Philippe': "M", 'Philippe-Alexandre': "M",
            'Philomène': "F", 'Pierre': "M", 'Pierre-Alexandre': "M", 'Pierre-André': "M", 'Pierre-Antoine': "M", 'Pierre-Côme': "M",
            'Pierre-Cyril': "M", 'Pierre-Cyrille': "M", 'Pierre-Emmanuel': "M", 'Pierre-Etienne': "M", 'Pierre-Eugène': "M", 'Pierre-Henri': "M",
            'Pierre-Jean': "M", 'Pierre-Julien': "M", 'Pierre-Louis': "M", 'Pierre-Marie': "M", 'Pierre-Olivier': "M", 'Pierre-Paul': "M",
            'Pierre-Quentin': "M", 'Pierrette': "F", 'Pierre-Valentin': "M", 'Pierre-Vincent': "M", 'Pierre-Xavier': "M", 'Pierre-Yves': "M",
            'Pierrick': "M", 'Pierrot': "M", 'Pietro': "M", 'Pilar': "F", 'Pol': "M", 'Primo': "M",
            'Prosper': "M", 'Rabah': "M", 'Rabia': "F", 'Rachel': "F", 'Rachid': "M", 'Rachida': "F",
            'Rahma': "F", 'Ralph': "M", 'Ramazan': "M", 'Ramdane': "M", 'Ramiro': "M", 'Ramon': "M",
            'Raoul': "M", 'Raphaël': "M", 'Raphaëlle': "F", 'Rayane': "M", 'Raymond': "M", 'Raymonde': "F",
            'Raynald': "M", 'Rebecca': "F", 'Regine': "F", 'Regis': "M", 'Reine': "F", 'Remi': "M",
            'Remy': "M", 'Renato': "M", 'Renaud': "M", 'René': "M", 'René-Charles': "M", 'René-Claude': "M",
            'Renée': "F", 'René-Jean': "M", 'René-Marc': "M", 'René-Paul': "M", 'René-Pierre': "M", 'René-Yves': "M",
            'Reynald': "M", 'Ricardo': "M", 'Richard': "M", 'Ridha': "F", 'Rina': "F", 'Rino': "M",
            'Rita': "F", 'Robert': "M", 'Roberte': "F", 'Roberto': "M", 'Robin': "M", 'Rocco': "M",
            'Roch': "M", 'Rodolphe': "M", 'Rodrigue': "M", 'Roger': "M", 'Roland': "M", 'Rolande': "F",
            'Rolf': "M", 'Rolland': "M", 'Romain': "M", 'Roman': "M", 'Romane': "F", 'Roméo': "M",
            'Romuald': "M", 'Ronald': "M", 'Ronan': "M", 'Rosa': "F", 'Rosalie': "F", 'Rosario': "F",
            'Rose': "F", 'Rose-Anne': "F", 'Roseline': "F", 'Roselyne': "F", 'Rosemonde': "F", 'Rosette': "F",
            'Rosine': "F", 'Rosita': "F", 'Roxane': "F", 'Rozenn': "F", 'Rudolf': "M", 'Rudy': "M",
            'Rui': "M", 'Ruth': "F", 'Ryan': "M", 'Saad': "M", 'Saadia': "F", 'Sabine': "F",
            'Sabrina': "F", 'Sadia': "F", 'Sadok': "M", 'Saïd': "M", 'Saïda': "F", 'Salah': "M",
            'Salem': "M", 'Saliha': "F", 'Salim': "M", 'Salima': "F", 'Salomon': "M", 'Salvador': "M",
            'Salvator': "M", 'Samba': "F", 'Sami': "M", 'Samia': "F", 'Samir': "M", 'Samira': "F",
            'Samuel': "M", 'Samy': "?", 'Sandra': "F", 'Sandrine': "F", 'Santiago': "M", 'Santo': "M",
            'Sara': "F", 'Sarah': "F", 'Sarah-Anne': "F", 'Sarah-Eve': "F", 'Sarah-Jane': "F", 'Sarah-Laure': "F",
            'Sarah-Line': "F", 'Sarah-Lise': "F", 'Sarah-Lou': "F", 'Sarah-Louise': "F", 'Sarah-Marie': "F", 'Sarah-Myriam': "F",
            'Sauveur': "M", 'Sébastien': "M", 'Séraphin': "M", 'Serge': "M", 'Sergine': "F", 'Sergio': "M",
            'Séverin': "M", 'Séverine': "F", 'Sidonie': "F", 'Siegfried': "M", 'Silvio': "M", 'Siméon': "M",
            'Simon': "M", 'Simone': "F", 'Simonne': "F", 'Slimane': "M", 'Smail': "M", 'Solange': "F",
            'Sonia': "F", 'Sophia': "F", 'Sophie': "F", 'Sophie-Anne': "F", 'Sophie-Caroline': "F", 'Soraya': "F",
            'Souad': "F", 'Stanis': "M", 'Stanislas': "M", 'Stanislawa': "F", 'Stella': "F", 'Stéphan': "M",
            'Stéphane': "?", 'Stéphanie': "F", 'Stephen': "M", 'Steve': "M", 'Steven': "M", 'Susan': "F",
            'Suzanne': "F", 'Suzette': "F", 'Suzy': "F", 'Sybille': "F", 'Sylvain': "M", 'Sylvaine': "F",
            'Sylvère': "M", 'Sylvestre': "M", 'Sylvette': "F", 'Sylvia': "F", 'Sylviane': "F", 'Sylvianne': "F",
            'Sylvie': "F", 'Sylvio': "M", 'Tahar': "M", 'Tanguy': "M", 'Tania': "F", 'Tatiana': "F",
            'Tayeb': "M", 'Théo': "M", 'Théodore': "M", 'Théophile': "M", 'Thérèse': "F", 'Thibault': "M",
            'Thierry': "M", 'Thomas': "M", 'Tom': "M", 'Toni-Joe': "M", 'Tony': "M", 'Tony-Joe': "M",
            'Toussaint': "M", 'Tristan': "M", 'Ugo': "M", 'Ulysse': "M", 'Umberto': "M", 'Urbain': "M",
            'Ursula': "F", 'Valentin': "M", 'Valentine': "F", 'Valère': "M", 'Valérie': "F", 'Valéry': "M",
            'Vanessa': "F", 'Véra': "F", 'Véronique': "F", 'Victoire': "F", 'Victor': "M", 'Victoria': "F",
            'Victorin': "M", 'Victorine': "F", 'Vincent': "M", 'Vincente': "F", 'Vincent-Xavier': "M", 'Violaine': "F",
            'Violette': "F", 'Virgile': "M", 'Virgilio': "M", 'Virginia': "F", 'Virginie': "F", 'Vital': "M",
            'Vito': "M", 'Vittorio': "M", 'Vivian': "M", 'Viviane': "F", 'Vladimir': "M", 'Werner': "M",
            'Wilfrid': "M", 'Wilfried': "M", 'William': "M", 'Williams': "M", 'Willy': "M", 'Wladimir': "M",
            'Wladislas': "M", 'Wladislaw': "M", 'Wolfgang': "M", 'Xavier': "M", 'Xavier-Alexandre': "M", 'Xavier-François': "M",
            'Xavier-Marie': "M", 'Xavier-Pierre': "M", 'Yahia': "F", 'Yamina': "F", 'Yanick': "M", 'Yanis': "M",
            'Yann': "M", 'Yann-Ber': "M", 'Yann-Éric': "M", 'Yann-Gaël': "M", 'Yannick': "M", 'Yann-Vari': "M",
            'Yann-Yves': "M", 'Yasmina': "F", 'Yasmine': "F", 'Yolaine': "F", 'Yolande': "F", 'Youcef': "M",
            'Youssef': "M", 'Yvan': "M", 'Yveline': "F", 'Yves': "M", 'Yves-Alain': "M", 'Yves-Alexandre': "M",
            'Yves-André': "M", 'Yves-Éric': "M", 'Yves-Henri': "M", 'Yves-Laurent': "M", 'Yves-Michel': "M", 'Yves-Olivier': "M",
            'Yves-Pierre': "M", 'Yvette': "F", 'Yvon': "M", 'Yvonne': "F", 'Yvonnick': "?", 'Zahia': "F",
            'Zahra': "F", 'Zineb': "M", 'Zoé': "F", 'Zohra': "F", 'Zora': "F", 'Zoubida': "F" }

# Apellatifs (mots de civilite')
motsCivilite = { 'M.': "M", 'm.': "M", 'Mr': "M", 'mr': "M", 'mr.': "M", 'Mr.': "M", 'mr.': "M", 'Mme': "F",
                 'mme': "F", 'Mlle': "F", 'mlle': "F", 'Monsieur': "M", 'monsieur': "M", 'Madame': "F",
                 'madame': "F", 'Mademoiselle': "F", 'mademoiselle': "F" }

# Mots specifiques designant des individus humains et impliquant le sexe (parfois de facon implicite => dans l'article)
motsIndividu = { 'femme': "F", 'dame': "F", 'homme': "M", 'fille': "F", 'fillette': "F", 'garçon': "M",
                 'enfant': "?", 'adolescent': "M", 'adolescente': "F", 'adulte': "?", 'trentenaire': "?", 'quadragénaire': "?",
                 'quinquagénaire': "?", 'sexagénaire': "?", 'septuagénaire': "?", 'octogénaire': "?", 'nonagénaire': "?", 'centenaire': "?",
                 'senior': "?", 'vieillard': "M", 'vieille': "F" }

# Formes de surface d'adjectifs ou de participes passe's s'appliquant a des etres humains
# et marques en genre
adjectifsHumain = { 'Âgé': "M", 'âgé': "M", 'Âgée': "F", 'âgée': "F", 'Né': "M", 'né': "M",
                    'Née': "F", 'née': "F", 'Jeune': "?", 'jeune': "?", 'Décédé': "M", 'décédé': "M",
                    'Décédée': "F", 'décédée': "F", 'Marié': "M", 'marié': "M", 'Mariée': "F", 'mariée': "F",
                    'Hospitalisé': "M", 'hospitalisé': "M", 'Hospitalisée': "F", 'hospitalisée': "F", 'Transféré': "M", 'transféré': "M",
                    'Transférée': "F", 'transférée': "F", 'Adressé': "M", 'adressé': "M", 'Adressée': "F", 'adressée': "F",
                    'Présenté': "M", 'présenté': "M", 'Présentée': "F", 'présentée': "F" }

# Idem pour cestains adjectifs ou participes passes tres caracteristiques de la
# description du patient dans les textes medicaux, mais necessitant un controle
# du contexte (... en consultation / ... dans le service ...)
adjectifsHumainEnContexteConsultation = { 'Vu': "M", 'vu': "M", 'Vue': "F", 'vue': "F", 'Venu': "M", 'venu': "M",
                                          'Venue': "F", 'venue': "F", 'Admis': "M", 'admis': "M", 'Admise': "F", 'admise': "F",
                                          'Revu': "M", 'revu': "M", 'Revue': "F", 'revue': "F", 'Revenu': "M", 'revenu': "M",
                                          'Revenue': "F", 'revenue': "F", 'Suivi': "M", 'suivi': "M", 'Suivie': "F", 'suivie': "F" }

# Prefixes rencontres au debut de termes anatomiques, biologiques, cliniques ou medicaux
# qui sont tres specifiques a l'un ou l'autre sexe
stemsBioMed = { 'Brenner': "F", 'IMG': "F", 'IVG': "F", 'PGP': "F", 'Fallope': "F", 'Graaf': "F",
                'Meigs': "F", 'accouch': "F", 'adénar': "M", 'amniotom': "F", 'aménorr': "F", 'andropause': "M",
                'anovulation': "F", 'areole': "F", 'aréole': "F", 'asperm': "M", 'asthénozoosperm': "M", 'avort': "F",
                'azoosperm': "M", 'balanite': "M", 'bourse': "M", 'caduques': "F", 'caverneux': "M", 'cervix': "F",
                'cesarienne': "F", 'clito': "F", 'colpo': "F", 'cryptorchi': "M", 'culdo': "F", 'cumulus': "F",
                'césarienne': "F", 'dysménorr': "F", 'déciduome': "F", 'déférent': "M", 'embryo': "F", 'enceinte': "F",
                'endometr': "F", 'endomètr': "F", 'foet': "F", 'gestation': "F", 'gland': "M", 'granulosa': "F",
                'grossesse': "F", 'gynatrésie': "F", 'gynéco': "F", 'hymen': "F", 'hydrocolpos': "F", 'hydrocèle': "M",
                'hymen': "F", 'hymén': "F", 'hyster': "F", 'hystér': "F", 'hématocolpos': "F", 'hématocèle': "M",
                'hémosperm': "M", 'intrathcécal': "F", 'lactation': "F", 'lactéa': "F", 'leucorrhée': "F", 'luteum': "F",
                'lutéal': "F", 'lutéin': "F", 'lutéo': "F", 'mamelon': "F", 'mamma': "F", 'mammo': "F",
                'maste': "F", 'menstru': "F", 'multipar': "F", 'myomètr': "F", 'myometr': "F", 'ménarc': "F",
                'ménarq': "F", 'ménopaus': "F", 'ménorr': "F", 'obstetr': "F", 'obstétr': "F", 'oestrad': "F",
                'oestro': "F", 'oligoménorr': "F", 'oligosperm': "M", 'oocyt': "F", 'oophor': "F", 'orchi': "M",
                'ovar': "F", 'ovaire': "F", 'oviduc': "F", 'ovul': "F", 'paraphimosis': "M", 'phall': "M",
                'phimosis': "M", 'postménopause': "F", 'postpartum': "F", 'priapisme': "M", 'primipar': "F", 'progest': "F",
                'prostat': "M", 'prémenstru': "F", 'préménopause': "F", 'prépuce': "M", 'pyométrie': "F", 'pélyco': "F",
                'pénie': "M", 'pénis': "M", 'périménopause': "F", 'péripartum': "F", 'rectovagin': "F", 'règles': "F",
                'salpin': "F", 'sein': "F", 'superovulat': "F", 'scrot': "M", 'sperm': "M", 'spongieux': "M",
                'séminal': "M", 'testi': "M", 'testo': "M", 'thcécal': "F", 'tocoly': "F", 'tumescen': "M",
                'tératozoosperm': "M", 'unipar': "F", 'uter': "F", 'utér': "F", 'vagin': "F", 'varicocèle': "M",
                'verge': "M", 'vulv': "F", 'vésicovagin': "F", 'éjacul': "M", 'épididym': "M", 'épisiotom': "F",
                'érecti': "M", '\u0153strad': "F", '\u0153stro': "F" }

myArgumentParser=argparse.ArgumentParser()
myArgumentParser.add_argument("datadir", help="data directory")
args=myArgumentParser.parse_args()

directory = args.datadir

# Petite fonction accessoire utile : trouver le genre du determinant d'un nom

def genreDeterminant(thisNoun, sentence):
    gender = None
    if (((thisNoun.__class__) == stanza.models.common.doc.Word) and ((sentence.__class__) == stanza.models.common.doc.Sentence)):
        if ((thisNoun.upos == "NOUN") or (thisNoun.upos == "PROPN")):
            for token in sentence.tokens:
                for word in token.words:
                    if ((word.id != thisNoun.id) and (word.head == thisNoun.id) and (word.deprel == "det") and (word.feats != None)):
                        features=(word.feats).split('|')
                        for feature in features:
                            [ parameter, value ] = feature.split('=')
                            if (parameter == 'Gender'):
                                gender = value
    return gender            

# Pour un fichier source, calcule un certain nombre de parametres
# d'entree qui ont l'air pertinents
# 1. pour le sexe (deux classes en sortie : M ou F)
# 2. pour l'age (cinq classes : nourrisson, enfant, adolescent, adulte, senior + age precis si disponible)
# (remarques plus detaillees sur le corpus d'apprentissage plus bas, en fin de fichier)

# Sexe :
# 1. genreMotsPatient : genre des occurrences du mot patient, s'il est present
# 2. genreMotsCivilite : genre des appellatifs de civilite (M, Mme...)
# 3. genreMotsIndividu : genre des termes designant un individu (homme, femme, fille, garcon...)
# 4. genreAdjectifsHumains : genre des adjectifs s'appliquant a des etres humains (hospitalise'(e), admis(e), a^ge'(e), ne'(e), marie'(e), enceinte, ...)
# 5. genrePrenoms : genre des prenoms
# 6. genreStemsBioMed : indications sur le sexe du patient apportees par des termes d'anatomie, de physiologie ou de pathologie specifiquement masculins ou feminins
# (NB.Choix arbitraire: compter les occurrences du feminin en >0
#  et les occurrences du masculin en <0)
# On fait decroitre le poids avec la distance au debut du texte

def dictParametresPertinentsSexe(annotatedDocument):
    relevantFeatures = { 'genreMotsPatient': 0.0, 'genreMotsCivilite': 0.0, 'genreMotsIndividu': 0.0, 'genreIndicationExpliciteDuSexe': 0.0, 'genreAdjectifsHumain': 0.0, 'genrePronoms': 0.0, 'genrePrenoms': 0.0, 'genreStemsBioMed': 0.0 }
    dictMotsDocument = { -1: "empty" }
    ns = len(annotatedDocument.sentences)
    lastWord = 0
    for s in range(ns):
        nt = len(annotatedDocument.sentences[s].tokens)
        for t in range(nt):
            nw = len(annotatedDocument.sentences[s].tokens[t].words)
            for w in range(nw):
                word = annotatedDocument.sentences[s].tokens[t].words[w]
                dictMotsDocument[lastWord] = { 'word': word, 'sentence': s }
                lastWord += 1
    dictMotsDocument.pop(-1)
    s = 0
    for w in range(lastWord):
        word = dictMotsDocument[w]['word']
        s = dictMotsDocument[w]['sentence']
        genderTag=None
        detGenderTag=None
        # Genre du mot tel qu'il a ete etiquete par le POS-tagger
        # (a prendre avec an peu de precaution parce qu'il y a parfois des erreurs d'etiquetage ;
        #  => la forme de surface, si elle est marquee en genre, est plus fiable)
        if (word.feats!=None):
            features=(word.feats).split('|')
            for feature in features:
                [ parameter, value ] = feature.split('=')
                if (parameter == 'Gender'):
                    genderTag = value
                if (parameter == 'Number'):
                    numberTag = value
                if (parameter == 'Person'):
                    personTag = value
                if (parameter == 'PronType'):
                    pronTypeTag = value
        # Genre du determinant dans le cas ou le mot est un nom
        if ((not (word.id==1)) and (word.upos in ["NOUN", "PROPN"])):
            detGenderTag = genreDeterminant(word,annotatedDocument.sentences[s])
        # ---- Genre du mot "patient(e)" ou "malade"
        if ((word.upos=="NOUN") and ((word.lemma=="patient") or (word.lemma=="malade"))):
            if (genderTag == 'Fem'):
                relevantFeatures['genreMotsPatient']+=(1-((s/ns)/(att)))
            elif (genderTag == 'Masc'):
                relevantFeatures['genreMotsPatient']-=(1-((s/ns)/(att)))
        # ---- Genre des appellatifs (mots de civilite)
        # pour "M." : verifications supplementaires parce que c'est parfois une initiale de nom propre
        if (word.text in ["m.", "M."]):
            if ((word.id==1) or (not ((dictMotsDocument[w-1]['word'].upos) in ["NOUN", "PROPN"]))) and (word.upos in ["NOUN", "PROPN"]):
                relevantFeatures['genreMotsCivilite']-=(1-((s/ns)/(att)))
        elif (word.text in motsCivilite.keys()):
            if (motsCivilite[word.text]=="F"):
                relevantFeatures['genreMotsCivilite']+=(1-((s/ns)/(att)))
            elif (motsCivilite[word.text]=="M"):
                relevantFeatures['genreMotsCivilite']-=(1-((s/ns)/(att)))
        # ---- Genre des mots qui designent les individus et qui impliquent une determination du sexe (homme, femme, garcon, fille...)
        if (word.lemma in motsIndividu.keys()):
            if (motsIndividu[word.lemma]=="F"):
                relevantFeatures['genreMotsIndividu']+=(1-((s/ns)/(att)))
            elif (motsIndividu[word.lemma]=="M"):
                relevantFeatures['genreMotsIndividu']-=(1-((s/ns)/(att)))
            # pour le terme "enfant", qui est epicene, on regarde le genre de son determinant
            else:
                if ((not (word.id==1)) and (word.upos in ["NOUN", "PROPN"])):
                    if (detGenderTag == "Fem"):
                        relevantFeatures['genreMotsIndividu']+=(1-((s/ns)/(att)))
                    elif (detGenderTag == "Masc"):
                        relevantFeatures['genreMotsIndividu']-=(1-((s/ns)/(att)))
        # ---- Mention explicite "sexe feminin" ou "sexe masculin"
        if (word.text == "sexe"):
            contextWord = dictMotsDocument[w+1]['word']
            if ((contextWord.head == word.id) and (contextWord.upos == "ADJ")):
                if (contextWord.text == "fémininin"):
                    relevantFeatures['genreIndicationExpliciteDuSexe']-=(1-((s/ns)/(att)))
                elif (contextWord.text == "masculin"):
                    relevantFeatures['genreIndicationExpliciteDuSexe']-=(1-((s/ns)/(att)))
        # ---- Genre des adjectifs qui s'appliquent a des etres humains et impliquent determination du genre
        if (word.text in adjectifsHumain.keys()):
            if (adjectifsHumain[word.text]=="F"):
                relevantFeatures['genreAdjectifsHumain']+=(1-((s/ns)/(att)))
            elif (adjectifsHumain[word.text]=="M"):
                relevantFeatures['genreAdjectifsHumain']-=(1-((s/ns)/(att)))
            # pour le terme "jeune", qui est epicene, on regarde le genre, en esperant qu'il a ete bien etiquete
            else:
                if (genderTag == 'Fem'):
                    relevantFeatures['genreAdjectifsHumain']+=(1-((s/ns)/(att)))
                elif (genderTag == 'Masc'):
                    relevantFeatures['genreAdjectifsHumain']-=(1-((s/ns)/(att)))
        # certains adjectifs/participes passes sont a interpreter en contexte ("... en consultation", "... dans le service" ...)
        if (word.text in adjectifsHumainEnContexteConsultation.keys()):
            d = 1
            while ((w+d<lastWord) and (dictMotsDocument[w+d]['sentence']==s)):
                contextWord = dictMotsDocument[w+d]['word']
                if ((contextWord.head == word.id) and (contextWord.lemma in ["cabinet", "consultation", "service", "urgence", "hôpital"])):
                    if (adjectifsHumainEnContexteConsultation[word.text]=="F"):
                        relevantFeatures['genreAdjectifsHumain']+=(1-((s/ns)/(att)))
                    elif (adjectifsHumainEnContexteConsultation[word.text]=="M"):
                        relevantFeatures['genreAdjectifsHumain']-=(1-((s/ns)/(att)))
                d+=1
        # ---- Genre des pronoms personnels 3ps
        if (word.upos == "PRON"):
            if ((pronTypeTag == "Prs") and (personTag=="3") and (numberTag=="Sing")):
                if (genderTag=="Fem"):
                    relevantFeatures['genrePronoms']+=(1-((s/ns)/(att)))
                elif (genderTag=="Masc"):
                    relevantFeatures['genrePronoms']-=(1-((s/ns)/(att)))
        # ---- Genre des prenoms
        if (word.text in prenoms.keys()):
            if (prenoms[word.text]=="F"):
                relevantFeatures['genrePrenoms']+=(1-((s/ns)/(att)))
            elif (prenoms[word.text]=="M"):
                relevantFeatures['genrePrenoms']-=(1-((s/ns)/(att)))
        # ---- Genre implique pas des stems specifiques du domaine biomedical (genre de la personne, pas genre du mot, bien sur)
        for stem in stemsBioMed.keys():
            if (word.text.startswith(stem)):
                if (stemsBioMed[stem]=="F"):
                    relevantFeatures['genreStemsBioMed']+=(1-((s/ns)/(att)))
                elif (stemsBioMed[stem]=="M"):
                    relevantFeatures['genreStemsBioMed']-=(1-((s/ns)/(att)))
    return relevantFeatures

# Calcul pour tous les fichiers du repertoire

def dumpSexFeaturesAsCSV(directory,CsvFileName):
    # tokenization, POS-tagging, extraction des indices morphologiques
    # et des dependances syntaxiques: delegue aux outils de Stanford
    # pour le francais (telecharger le paquet stanza et le modele 'fr')
    pipeline = stanza.Pipeline('fr')
    CsvFileHandler = open(CsvFileName, 'w', encoding='utf8')
    print("case,genreMotsPatient,genreMotsCivilite,genreMotsIndividu,genreIndicationExpliciteDuSexe,genreAdjectifsHumain,genrePronoms,genrePrenoms,genreStemsBioMed", file=CsvFileHandler)
    with os.scandir(directory) as entries:
        for entry in entries:
            if ((not (entry.name.startswith('.'))) and (entry.is_file()) and (entry.name.endswith('.txt'))):
                documentFileName = entry.name
                documentFileHandler = open(directory+'/'+documentFileName, 'r', encoding='utf8')
                documentTextContent = documentFileHandler.read()
                annotatedDocument = pipeline(documentTextContent)

                # Liste des parametres qui ont l'air pertinents pour determiner le sexe et l'age
                # (remarques plus detaillees sur le corpus d'apprentissage plus bas, en fin de fichier)
                features = dictParametresPertinentsSexe(annotatedDocument)
                print(documentFileName+",", end='', file=CsvFileHandler)
                print(str(features['genreMotsPatient'])+",", end='', file=CsvFileHandler)
                print(str(features['genreMotsCivilite'])+",", end='', file=CsvFileHandler)
                print(str(features['genreMotsIndividu'])+",", end='', file=CsvFileHandler)
                print(str(features['genreIndicationExpliciteDuSexe'])+",", end='', file=CsvFileHandler)
                print(str(features['genreAdjectifsHumain'])+",", end='', file=CsvFileHandler)
                print(str(features['genrePronoms'])+",", end='', file=CsvFileHandler)
                print(str(features['genrePrenoms'])+",", end='', file=CsvFileHandler)
                print(str(features['genreStemsBioMed']), file=CsvFileHandler)
                return 0

def trainSexModel():
    # A faire une fois sur le corpus d'apprentissage
    # dumpFeaturesAsCSV('data/release/train2021-train', 'data/work/patient_sex_features.csv')

    # Recuperation des parametres de sexe du corpus d'apprentissage dans le fichier CSV

    CsvFileHandler = open('data/work/patient_sex_features_train.csv', 'r', encoding='utf8')
    sexFeatures = pandas.read_csv(CsvFileHandler, index_col=0)
    CsvFileHandler.close()

    # Recuperation de l'information de classe sexe dans le fichier CSV

    CsvFileHandler = open('data/work/patient_classes_train.csv', 'r', encoding='utf8')
    sexClasses=pandas.read_csv(CsvFileHandler,index_col=0,usecols=['case','sex'])
    CsvFileHandler.close()

    # On entraine un classifieur AdaBoost

    clf = AdaBoostClassifier(n_estimators=400, random_state=0)
    X=sexFeatures.to_numpy()
    y=(sexClasses.to_numpy()).ravel()
    clf.fit(X,y)

    # On enregistre les parametres du modele
    clfLearnedModelFile = open('data/work/sex_adaboost_clf.pickle', 'wb')
    pickle.dump(clf, clfLearnedModelFile)
    clfLearnedModelFile.close()
    return 0

def classifySex(classifier, pipeline, documentFileName):
    documentFileHandler = open(documentFileName, 'r', encoding='utf8')
    documentTextContent = documentFileHandler.read()
    annotatedDocument = pipeline(documentTextContent)
    features = dictParametresPertinentsSexe(annotatedDocument)
    featuresArray = [ [ features['genreMotsPatient'], features['genreMotsCivilite'], features['genreMotsIndividu'], features['genreIndicationExpliciteDuSexe'], features['genreAdjectifsHumain'], features['genrePronoms'], features['genrePrenoms'], features['genreStemsBioMed'] ] ]
    return (classifier.predict(featuresArray))[0]

# =============================
# Un petit tour pour evaluation

trainSexModel()
clfLearnedModelFile = open('data/work/sex_adaboost_clf.pickle', 'rb')
clf = pickle.load(clfLearnedModelFile)

pipeline = stanza.Pipeline('fr')
print("")

with os.scandir(directory) as entries:
    for entry in entries:
        if ((not (entry.name.startswith('.'))) and (entry.is_file()) and (entry.name.endswith('.txt'))):
            sexClass = classifySex(clf, pipeline, directory+"/"+entry.name)
            print("\""+entry.name+"\",", end='')
            print("\""+sexClass+"\"")

print("")

# ==================================================================================================================
# 
# SEXE ET AGE
# information pertinente presque toujours dans la premiere phrase
# 
# ------------------------------------------------------------------------------------------------------------------
# 
# SEXE
# patient, patiente, malade (en verifiant le genre de l'article)
# Mme, Mlle, M., Mr., Madame, Monsieur
# homme, femme, fille, garc,on
# genre des determinants pour les noms epicenes (UNE enfant, CETTE enfant [filepdf-119-cas])
#    Attention nom epicene parfois utilise avec an article masculin generique meme
#    si le sujet est de sexe feminin (ex. "Un cadre de 42 ans" [filepdf-496-1-cas.txt])
# (-) genre de l'attribut du sujet pour les noms epicenes
# genre des pronoms 3ps
#    NB. sans identification des coreferences ce n'est pas tres pertinent
#    NB. sans identification des coreferences ce n'est pas tres pertinent
#    ("elle" renvoie souvent a un referent non-humain (ex. une tumeur), et "il",
#    en plus, apparait souvent dans "il s'agit" et autres expressions impersonnelles)
# genre des adjectifs/participes 3ps qui s'appliquent a des humains
#    (ne'(e), a^ge'(e), marie'(e), enceinte)
#    (si info disponible des adjectifs de'pendant du sujet [patient])
# NB. verifier la forme de surface est plus fiable que verifier le lemme + le genre,
#     car il y a parfois des erreurs d'etiquetage (ex. "revue" participe passe feminin
#     etiquete comme un nom avec le lemme "revue")
# participes passes frequemment utilises dans les CR medicaux pour decrire
# le contexte d'admission :
#    admis, transfe're', hospitalise', pre'sente', vu, revu, suivi
#    (dans le cas de mots tres generaux comme "vu", "revu" ou "suivi",
#     il faut verifier qu'ils sont dans le contexte "... en consultation"
#     ou "... aux urgences" ou "... dans le service", sinon ils peuvent
#     s'appliquer a autre chose que des humains)
# prenoms typiquement genre's, ex. "Christophe" [filepdf-554-5-cas.txt]
#    (parfois avec determinant, ex. "le jeune Flavian" [filepdf-513-cas.txt])
# "sexe masculin", "sexe fe'minin" : on en trouve des mentions explicites mais elles
# sont finalement relativement peu significatives (en effet elles apparaissent parfois dans
#    des contextes ou on ne parle pas du patient, p.ex. "mere d'un enfant de sexe masculin",
#    "femme exercant dans un milieu professionnel majoritairement masculin"...)
# organes specifiques a un sexe: pe'nis, verge, testicule, prostate, ute'rus, vagin, scrotum, phallus, bourse, pre'puce, sein, vulve
# adjectifs: caverneux, mammaire
# morphemes associes: pe'n-, testic-, orchid-, mamm-, hyster-, scrot-, spermat-, vagin-, vulv-
# evenements/sosy specifiques a un sexe: grossesse, re`gles, ame'norrhe'e, me'nopause, accouchement
# noms de specialite: gyne'cologue
# "pere de (...) enfant(s)" / "mere de (...) enfant(s)"
# "mari": le mot n'apparait que dans des documents concernant des femmes
# tres rarement: sexe non mentionne :
# (1) parfois parce que le texte parle de PLUSIEURS patients (filepdf-417-cas);
# (2) sinon, lorsque le texte parle bien d'un individu, mais qu'on ne trouve aucun indice
# mentionnant le sexe (filepdf-63-cas), c'est que cela n'a aucune importance pour la maladie
# 
# ------------------------------------------------------------------------------------------------------------------
# 
# AGE
# la plupart du temps sous la forme: "58 ans", "17 mois"
#    frequemment precede de "a^ge' de ..." ou simplement dans le contexte "un homme de 44 ans"
#    l'age du patient semble etre toujours dans la premiere phrase
# une expression du type "5 ans" peut aussi parfois etre un delai ("e'volution favorable avec un recul de 5 ans" [filepdf-86-4-cas.txt])
#    parfois on a les deux dans la premiere phrase (ex. filepdf-517-1-cas.txt)
# rarement, en lettres: "trente ans"
# tres rarement: annee de naissance ("ne'e en 1923" [filepdf-49-cas]) - l'age peut etre deduit si on connait la date de redaction du document
#     (c'est parfois difficile a deduire - dans l'exemple filepdf-49-cas, le document mentionne une date
#      d'operation en 1999, puis se termine par "la surveillance clinique a` un an n'a pas montre' de me'tastase" ...)
# parfois: classes d'age: garc,on, fille, fillette [filepdf-91-3-cas], enfant, nourrisson [filepdf-116-cas]
# rarement: pas d'age mais classe d'age (nourrisson, enfant, adolescent, adulte, senior, vingtaine d'annees, cinquantaine)
# 
# ------------------------------------------------------------------------------------------------------------------
# 
# CAS PARTICULIER
# la description de cas concerne plusieurs individus:
# filepdf-417-cas, filepdf-533-1-cas.txt, filepdf-554-2-cas.txt (~= filepdf-533-1-cas.txt)
# 
# ==================================================================================================================
