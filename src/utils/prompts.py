EXAMPLES_TEMPLATE = '''
We are interested in extracting event information from spanish legal text and you are a system that will compute that task for us.

Given a large set of sentences in spanish from the legal domain, written between triple backticks, your objective is to extract the roles from the text written in Spanish following the next steps and the examples of each step given between triple dashes:
1. Identify each sentence in the corpus separated for new lines.
2. In each sentence detect a subject entity, an object entity, an event trigger and a complement entity following the next definitions.
    * subject: Agent of the action, who performs the action.
    * event trigger: Action
    * object: Receiver of the action. There can be more than one object in the sentence.
    * complement: Item which is handled in the relation. There can be more than one complement in the sentence.

---
    %(role_example)s
---

3. Classify each subject in one of the following labels:
    * LegalAgent: Natural person
    * LegalEntity: Not natural person nor individual. Normally a corporation or an enterprise
    * LegalConcept: Not natural person nor corporation.

---
    %(subject_type)s
---

4. Classify each object entity in one of the labels of previous step. If no object is present in the sentence, just leave it as null.

---
    %(object_type)s
---

5. Classify the relation in one of the following classes:
    * Right
    * Duty
    * NoRight
    * Privilege
Right and No-Right are opposites and Duty and Priviledge are opposite as well. If a sentence does not contains a relation classify it as Norelation.

---
    %(relation_type)s
---

6. The output of the task should be a list of dictionaries, each dictionary contains the following keys:
    * text: the sentence
    * subject: the subject entity
    * object: the list of object entities or empty list
    * complement: the list of complement entities or empty list
    * event: the event trigger
    * subjectLabel: the classification of the subject entity in step 3
    * objectLabel: the classification of the object entity in step 4.  All objects entities have the same classification
    * relationType: the classification of the relation in step 5
'''
