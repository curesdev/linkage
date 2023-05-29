# Protocol

In this document we present a procotol for tokenizing and linking personal identifiable information.


# Tokenization of PII

## Supported PII

### Name

* First name: string
* Middle name: string
* Last name: string

* Full name: string

* 1st initial of first name: string
* First 3 letters of first name: string
* 1st initial of last name: string
* First 3 letters of last name: string

* First name soundex: string
* Last name soundex: string

### Sex

* Sex: categorical

### Race

* Race: categorical

### Location at birth

* City: string
* State: string
* Zip code: number
* 3 digit Zip code: number
* Address first line: string

### IDs

* SSN: string (number)
* Government issued ID or national ID: string

### Parents' data

* Father first name: string
* Father last name: string
* Father date of birth: string
* Father full name: string

* Mother first name: string
* Mother last name: string
* Mother date of birth: string
* Mother full name: string

### Contact information

* Email: string
* Phone number: string (number)

## Tokenization

Each field is tokenized separately following the same process:

1. Normalization
2. Expansion
3. Instertion in a Bloom filter
4. Bloom filter is diffused with differential privacy

### Normalization

The following rules are applied to each field:

* Remove whitespace: "J M Smith " -> "JMSmith".
* Remove punctuation: "Víctor" -> "Victor".
* Convert to lowercase: "Jonathan" -> "jonathan".
* Only letters/numbers are keps for names/numerical fields.
* Categorical fields (Sex and Race) are converted to a categorical value.

### Expansion

In order to account for inaccuracies in the data, we expand each value so that similar values generate similar tokens.

#### Bigrams

All string fields are expanded into bigrams. For example:

* j -> j
* jo -> jo
* peter -> pe, et, te, er

#### Date

Each date is expanded to a list of dates:

* The same day
* The day before
* The day after
* The first day of the month
* The first day of the year

### Bloom filters

Each field is encoded as a Bloom filter of **1024** bits.

We use a dynamic number of hash functions so that the number of bits inserted into the Bloom filter is **200**.

### Differential privacy

Differential privacy with $\epsilon=2$ is applied to the Bloom filter, that is, each bit of the Bloom filter is flipped with probability:

$$P_\text{flip}=\frac{1}{1 + e^\epsilon}$$


# Linking

