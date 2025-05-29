The goal is to evaluate the accuracy of the _inflecteur_ Python module transformations.

To do this, we used the same sentences used for the generation/error type evaluation, and extracted the dependencies to be changed by inflecteur (`original_dep`). We also tracked supplemental changes made by GeNRe (in particular, to fix past participles). This gives us two additional keys: `inflecteur_replacement` and `genre_replacement`.

`original_sent` refers to the original, non-inclusive sentence to be changed, and `converted_sent` is the final transformed sentence by GeNRe.

The value associated to the `manual` key should be manually changed to indicate the correct, final version of the dependency based on `converted_sent`.

For example:

```json
{
  "eupr_24": [
    {
      "original_sent": "Big Brother Europe est en marche, mais apparemment, personne ne doit le savoir, même pas les députés européens.",
      "converted_sent": "Big Brother Europe est en marche, mais apparemment, personne ne doit le savoir, même pas le parlement européen.",
      "original_dep": "européens",
      "inflecteur_replacement": "européen",
      "genre_replacement": "européen",
      "manual": "UNK_MANUAL"
    }
  ]
}
```

`original_dep` refers to the dependency found in `original_sent`. `inflecteur_replacement` refers to the new inflection made by inflecteur. `genre_replacement` refers to any additional replacement made by GeNRe (notably for past participles; here, there are no changes compared to inflecteur's output).

The value for the key `manual` (UNK_MANUAL) should be replaced with the correct inflection of the dependency AFTER the member noun → collective noun replacement. This means that we should indicate "européen" here (to reflect agreement with "parlement"). The main key `eupr_24` should thus look like this:

```json
{
  "eupr_24": [
    {
      "original_sent": "Big Brother Europe est en marche, mais apparemment, personne ne doit le savoir, même pas les députés européens.",
      "converted_sent": "Big Brother Europe est en marche, mais apparemment, personne ne doit le savoir, même pas le parlement européen.",
      "original_dep": "européens",
      "inflecteur_replacement": "européen",
      "genre_replacement": "européen",
      "manual": "européen"
    }
  ]
}
```

In total, there are 401 dependencies to annotate (196 for Europarl; 205 for Wikipedia). They are available in two distinct files: `inflecteur_eval_eupr_gen_eval` (for Europarl) and `inflecteur_eval_wiki_gen_eval` (for Wikipedia).

ONLY the values of key `manual` should be changed, everything else should be kept as is.