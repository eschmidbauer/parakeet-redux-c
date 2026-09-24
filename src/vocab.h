#ifndef PK_VOCAB_H
#define PK_VOCAB_H

#include "pk.h"

typedef struct {
    char *raw;         /* the piece as written in vocab.txt */
    char *text;        /* the piece with U+2581 replaced by a space */
    int skip;          /* <unk>, <pad>, <blk> */
    int word_start;    /* raw starts with U+2581 */
    int is_punct;      /* text.strip() is non-empty and all punctuation */
} pk_piece;

typedef struct {
    pk_piece *pieces;
    int n;
    int blank_id;
} pk_vocab;

int pk_vocab_load(pk_vocab *v, const char *path);
/* The same from the text of vocab.txt in memory. */
void pk_vocab_parse(pk_vocab *v, const char *text, size_t len);
void pk_vocab_free(pk_vocab *v);

/* SentencePiece detokenization (one leading space dropped). malloc'd. */
char *pk_vocab_decode(const pk_vocab *v, const int *ids, int n);
/* Word timestamps from the transducer's token durations (Photon's rule). */
int pk_vocab_words(const pk_vocab *v, const int *ids, const int *durations, int n, double frame_seconds, pk_word **words);

/* Python-style str.strip() of ASCII and common Unicode whitespace, in place; returns s. */
char *pk_strip(char *s);
/* True if the word, right-stripped, ends with '.', '?' or '!'. */
int pk_ends_sentence(const char *word);

#endif
