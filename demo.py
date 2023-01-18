from lib import SemanticSentenceSimilarity

if __name__ == '__main__':
    sentences = ["Seni seviyorum.",
                 "Pizza mı seversin makarna mı?",
                 "Pizza ve makarna yedim.",
                 "I ate dinner.",
                 "We had a three-course meal.",
                 "Brad came to dinner with us.",
                 "He loves fish tacos.",
                 "In the end, we all felt like we ate too much.",
                 "We all agreed; it was a magnificent evening.",
                 "Sağ kol omzuna değdir.",
                 "Sol el açık düz, karın hizasında avuç içi aşağı bakacak biçimdedir.",
                 "Sağ el karın hizasında, avuç içi yukarı bakacak şelide açıktır.",
                 "Sağ el açık düz, karın hizasında avuç içi aşağı bakacak biçimdedir."
                 ]
    extra_sentences = ["Bugün multu hissetmiyorum.", "Amcanlara selam söyle."]

    # query_sentence = "Senden hoşlanıyorum."
    query_sentence = "Sağ el açık düz, karın hizasında avuç içi yukarı bakacak biçimdedir."
    query_sentence = "Every body thinks this evening was good."

    # sss = SemanticSentenceSimilarity(model_name_or_path='distiluse-base-multilingual-cased-v1', device="cuda")
    # sss = SemanticSentenceSimilarity(model_name_or_path='best_model_pooling_512', device="cuda")
    sss = SemanticSentenceSimilarity(model_name_or_path='best_model_pooling_768_pca_512', device="cuda")
    # sss.extend_db(sentences)
    # sss.extend_db(extra_sentences)
    # print(sss.search_db(query_sentence))

    for i in sentences:
        print(i)
        print(sss.compare(sentence_one=query_sentence,
                          sentence_two=i))
