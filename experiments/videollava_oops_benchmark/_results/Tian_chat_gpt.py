import openai
import json
import re
import time
from tqdm import tqdm
import os
# from imagenet_labels import clip_imagenet_classes
from prompts import class_map, cub200_classes

# one_lengths = json.load(open('./one-lengths.json', 'r'))

openai.api_key = ''

def rename(label, dataset_name, definition, similar_names):
    # joined_text = ", ".join(similar_names[:-1])
    # joined_text += ' and ' + similar_names[-1]
    # print(joined_text)

    if dataset_name == 'dtd':
        return f'''
            What are some common ways of referring to a texture i.e. {label} ? Give me a numbered list only. Don't give any other text.
        '''
    
    elif dataset_name == 'fgvc-aircraft-2013b-variants102':
        return f'''
            What are some common ways of referring to an aircraft i.e. {label} ? Give me a numbered list only. Don't give any other text.
        '''
    
    elif dataset_name == 'oxford-flower-102':
        return f'''
            What are some common ways of referring to a flower i.e. {label} ? Give me a numbered list only. Don't give any other text.
        '''
    
    elif dataset_name == 'oxford-iiit-pets':
        return f'''
            What are some common ways of referring to a pet i.e. {label} ? Give me a numbered list only. Don't give any other text.
        '''

    elif dataset_name == 'sun397':
        return f'''
            What are some common ways of referring to a scene i.e. {label} ? Give me a numbered list only. Don't give any other text.
        '''

    elif dataset_name == 'food-101':
        return f'''
            What are some common ways of referring to a food i.e. {label} ? Give me a numbered list only. Don't give any other text.
        '''

    elif dataset_name == 'stanford-cars':
        return f'''
            What are some common ways of referring to a car i.e. {label} ? Give me a numbered list only. Don't give any other text.
        '''
    
    elif dataset_name == 'caltech-101':
        assert definition is not None
        return f'''
            What are some common ways of referring to a {label}, which is defined as {definition} ? Give me a numbered list only. Don't give any other text.
        '''

    else: # eruosat_clip
        return f'''
            What are some common ways of referring to a {label} ? Give me a numbered list only. Don't give any other text.
        '''
    
def timeout(t=30):
    for _ in tqdm(range(t), desc='waiting'):
        time.sleep(1)

def get_description(prompt_fn = rename, text='', dataset_name='', definition=None, extras=[]):
    description = ''
    isException = False
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful and honest assistant."},
                {"role": "user", "content": prompt_fn(text, dataset_name, definition, extras)},
            ]
        )
        # print('response', response)
        description = response['choices'][0]['message']['content']
    except:
        isException =True
        raise Exception(f'OpenAI API call failed. classname: {text}')
    
    return isException, description

def get_more_labels(extra_names):
    start_from = len(extra_names.items())
    print(start_from)
    for i, key in enumerate(clip_imagenet_classes):
        if i< start_from: continue
        print(f'Starting {i}')
        label = key #one_lengths[key]
        _, names = get_description(text=label)
        extra_names[key] = names.split('\n')
        if i % 40 == 0 and i>0:
            print(f'done till: {i}.')
            timeout(20)
        with open('./extras.json', 'w') as f:
            f.write(json.dumps(extra_names, indent=4))
    with open('./extras.json', 'w') as f:
        f.write(json.dumps(extra_names, indent=4))

def get_non_ambigous_names(extra_names, similar_classes):
    start_from = len(extra_names.items())
    print(start_from)
    for i, key in enumerate(clip_imagenet_classes):
        if i< start_from: continue
        print(f'Starting {i}')
        label = key #one_lengths[key]
        _, names = get_description(text=label, extras=similar_classes[str(i)]['similar_classes'])
        extra_names[i] = {'official':key, 'alternates':names.split('\n')}
        if i % 40 == 0 and i>0:
            print(f'done till: {i}.')
            timeout(20)
        with open('./extras-non-ambi.json', 'w') as f:
            f.write(json.dumps(extra_names, indent=4))
    with open('./extras-non-ambi.json', 'w') as f:
        f.write(json.dumps(extra_names, indent=4))

def clean_brackets(text):
    return re.sub(r'\([^)]*\)|\[.*?\]','', text)

def clean_text(text: str):
    return text.strip().replace("'",'').replace('"','').replace('-', ' ').replace('_', ' ').lower()

def combine_metrics_names():
    extra_names = json.load(open('./extra-names-filtered.json', 'r'))
    metrics = json.load(open('./metrics-filtered.json', 'r'))
    for i, key in enumerate(metrics):
        metrics[key]['name'] = extra_names[key]['name']
        metrics[key]['max_freq'] = ''

        if not clean_text(extra_names[key]['name']) in metrics[key]['alternates']:
            metrics[key]['alternates'][clean_text(extra_names[key]['name'])] = 0

        for name in extra_names[key]['alternates']:
            if not clean_text(name) in metrics[key]['alternates']:
               metrics[key]['alternates'][clean_text(name)] = 0

    with open('extras-metrics.json','w') as f:
        f.write(json.dumps(metrics, indent=4))


def get_alternative_names(class_lst, dataset_name):

    start_time = time.time()
    definition_lst = [None] * len(class_lst)
    # get the corresponding definition of the classname, only for caltech-101
    if dataset_name == 'caltech-101':
        definition = json.load(open('GPT3_caltech-101.tsv', 'r'))
        print('len(definition): ', len(definition))
        assert len(definition) == len(class_lst)

        # get the definition_lst
        definition_lst = [item["gpt3"][0] for item in definition]
        print('definition_lst[:3]: ', definition_lst[:3])

    # query the openai api for each class
    result_dict= {}
    start_from = 0
    #load the previously saved results if they exist
    if os.path.exists(f'output/{dataset_name}_alternatives.json'):
        result_dict = json.load(open(f'output/{dataset_name}_alternatives.json', 'r'))
        start_from = len(result_dict)
    
    print('start_from: ', start_from)
    for i, class_name in enumerate(class_lst):
        if i < start_from: continue # skip the previously obtained classes

        if name == 'semi-aves' or name == 'cub200':
            sname, cname = class_name
            class_name = cname

        _, alt_names = get_description(text=class_name, 
                                       dataset_name=dataset_name,
                                       definition=definition_lst[i])
        alt_names = alt_names.split('\n')
        
        print(f'\n{i}, class_name: {class_name}')
        print('alt_names:', alt_names)
        
        alt_names = [clean_text(clean_brackets(text.replace(f'{i + 1}.','').strip())) for i, text in enumerate(alt_names)]
        print('cleaned alt_names:', alt_names)

        # remove duplicated names
        alt_names = set(alt_names)
        
        # add class_name to the set if it's not already in the set
        if class_name not in alt_names:
            alt_names.add(class_name)            
        
        # for semi-aves and cub200, add the scientific name to the set
        if name == 'semi-aves' or name == 'cub200':
            alt_names.add(sname)

        # convert back to list
        alt_names = list(alt_names)
        # print('alt_names:', alt_names)

        # add the alt_names to the result_dict
        altername_names_dict = {}
        for alt_name in alt_names:
            altername_names_dict[alt_name] = 0
        result_dict[str(i)] = {'label': class_name, 'alternatives': altername_names_dict}

        # save the result_dict to a json file every 5 iterations
        if (i % 5 == 0 and i>0) or (i == len(class_lst)-1):
            with open(f'output/{dataset_name}_alternatives.json', 'w') as f:
                f.write(json.dumps(result_dict, indent=4))
            print(f'Saved output/{dataset_name}_alternatives.json')

        if i % 40 == 0 and i>0:
            print(f'done till: {i}.')
            timeout(90)
    
    print(f'Finished {dataset_name} in {round(time.time() - start_time)} seconds.')

def get_aves_class_lst():
    class_lst = []
    with open('../data/semi-aves/prompts/c-names_prompts.json', 'r') as f:
        data = json.load(f)

    for key, info in data.items():
        class_lst.append((info['species'], info['common_name']))

    return class_lst

def get_cub200_class_lst():
    class_lst = []
    for sname, cname in cub200_classes.items():
        class_lst.append((sname, cname))

    return class_lst



if __name__ == '__main__':

    # make output/ folder if it doesn't exist    
    if not os.path.exists('output'):
        os.makedirs('output')
        print('Created output/ folder.')

    # build the target dataset dict by picking items from the class_map dict
    target_dataset_dict = {}
    targets = [
            'imagenet-1k'
            # 'caltech-101', 
            # 'dtd', 'eurosat_clip', 'fgvc-aircraft-2013b-variants102',
            # 'oxford-flower-102', 'oxford-iiit-pets', 'sun397', 
            # 'food-101', 
            # 'stanford-cars'
            # 'semi-aves',
            # 'cub200'
            ]
    
    for key in targets:
        if key == 'semi-aves':
            target_dataset_dict[key] = get_aves_class_lst()
        elif key == 'cub200':
            target_dataset_dict[key] = get_cub200_class_lst()
        else: # for other datasets 
            target_dataset_dict[key] = class_map[key]

    print('len(target_dataset_dict): ', len(target_dataset_dict))
    
    # loop through each target dataset and run the procedure
    for name, class_lst in target_dataset_dict.items():
        print(f'\n{name}: {len(class_lst)}')
        get_alternative_names(class_lst, name) # +++++







