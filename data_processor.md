# Methods for deriving FCM edge weights from qualitative inputs.

## Constructing a Fuzzy Cognitive Map (Retrived from {})

<div align='justify'>

Once the structure of the FCM is available, we proceed with the second model-building step in which we need to identify the value of each edge. This is often done by providing a questionnaire to participants (Gray, Hilsberg, McFall, & Arlinghaus, 2015; Firmansyah, Supangkat, Arman, & Giabbanelli, 2019; Giabbanelli & Crutzen, 2014; Giabbanelli, Torsney-Weir, & Mago, 2012), who occasionally consist of (or are supplemented by) researchers leading the study (Mago, et al., 2013; Papada, Katsoulakos, Doulos, Kaliampakos, & Damigos, 2019). 
Then, a common practice consists of counting the fraction of respondents for each membership function (e.g. 3 out of 6 participants say “low”), clipping the function accordingly (i.e. truncate the triangle “low” after 3/6 of its height), and combining the functions and finding their center. We now detail this process on the example illustrated in Figure 1.

<img src="figures\figure_1.jpg" alt="figure not found" style="float: left; margin-right: 10px;" />
Figure 4: Four steps process to obtain the quantitative value of an edge’s causal strength from the qualitative evaluation of a panel of participants.

First, a questionnaire is provided, which asks participant to choose one linguistic variable for each relation (or subset of relations with which the participant is familiar). For instance, participants must evaluate the perceived causal impact of “perceived norms about sex and condom use” onto “belief that condoms must be used during sexual intercourse”. Once participants have completed the evaluation, their answers are aggregated. Here, we consider 6 participants: three chose ‘Medium’, one chose ‘Strong’, and two chose ‘Very Strong’. Then, we can use fuzzy logic per se. We define a membership function for each linguistic variable, allowing for overlaps between functions. As shown with triangular membership functions in Figure 3, there is a small possibility that a participant saying “medium” may think the same as one stating “high”. Fuzzy implications are used to take each membership function to the extent in which it was endorsed by participants. For instance, if 3 out of 6 participants (i.e. half) stated “medium”, then we project from the y-axis at 0.5 onto the membership function and preserve only the part under 0.5. This defines an implication as the minimum of the function that is triggered. After each membership function has been dealt with, we combine them to represent the judgment from the collective of participants. The combination consists of aggregating the function (i.e. taking the maximum) and using the centroid as representative value. Finally, as exemplified in Figure 4, the centroid is projected onto the fuzzy range of [0, 1] to obtain the strength of causation.

Although the fuzzy range produced by the process is always [0, 1], causal connections are not necessarily positive: there are also negative causations in which case some determinants decrease the value of others (Figure 3; red edges). The polarity of the causal connection is applied at the end of this process. For instance, the questionnaire in Figure 4 asked participants to predict the strength of the increase (Figure 1; Step 1) hence the value will be positive. Conversely, if participants had to predict the strength of the decrease, the result would be made negative (i.e. multiplied by -1). 
Note that participants may be given options such as “Non-existent” (Figure 1) or “I am not sure” in the questionnaire. Answers equivalent to “I am not sure” are skipped and thus not counted as part of the total number of respondents when computing the ratio of answers (Step 2→3, Figure 1). Answers equivalent to “Non-existent” can be mapped to 0 or a very narrow membership function (e.g. peaking at 0 and minimum at 0.001). The effect of introducing such an option is that some edges may turn out to carry a very small weight and may be dropped from the model.

This conversion process from qualitative linguistic terms into quantitative outputs is grounded in Fuzzy Set Theory, hence the name of ‘Fuzzy’ cognitive maps. As stated by Li and Zhang (emphases added), “Fuzzy set theory resembles human reasoning under approximate information and inaccurate data to generate decisions under uncertain environments. It is designed to mathematically represent uncertainty and vagueness, and to provide formalized tools for dealing with imprecision in real-world problems.” (Li & Zhang, 2007). The use of Fuzzy Set Theory distinguishes FCMs from conceptual maps (e.g. mind maps, concept maps) which either do not have causal weights or assign them through processes that are not mathematically designed to handle uncertainty. 

Given the use of fuzzy logic, the result from the process is a number in the interval [-1, 1] where -1 is the strongest negative causation and 1 the strongest positive causation. For instance, Figure 1 depicts a strong positive causation.
</div>

## Methods

<div align='justify'>

The methods presented in this section are used to derive the edge weights based on qualitative inputs as described above.

- [FcmDataProcessor](#FcmDataProcessor)
- [read_xlsx](#read_xlsx)
- [automf](#automf)
- [activate](#activate)
- [aggregate](#aggregate)
- [defuzzify](#defuzzify)
- [gen_weights_mat](#gen_weights_mat)
- [gen_weights_list](#gen_weights_list)
- [create_system](#create_system)

</div>

## FcmDataProcessor

To create an instance of FcmDataProcessor class you can either pass a dataframe that contains the data directly to the constructor 

```
fcm = FcmDataProcessor(df)
```
or use the read_xlsx method. 

## read_xlsx()
<div align='justify'>


```
read_xlsx(file_name, dtype)
```
</div>

## automf()
automf(universe, 
            linguistic_terms = ['-VH', '-H', '-M', '-L','-VL', 'VL','L', 'M', 'H', 'VH'])

## activate()
activate(activation_input, mf)

## aggregate()
aggregate(activated)

## defuzzify()
defuzzify(universe, aggregated, method = 'centroid')

## gen_weights_mat()
gen_weights_mat(data = None,
                    linguistic_terms = ['-VH', '-H', '-M', '-L', '-VL', 'VL','L', 'M', 'H', 'VH'],
                    method = 'centroid')

## gen_weights_list()
gen_weights_list(data = None,
                    linguistic_terms = ['-VH', '-H', '-M', '-L', '-VL', 'VL','L', 'M', 'H', 'VH'],
                    method = 'centroid') 

## create_system()
create_system(causal_weights = None)