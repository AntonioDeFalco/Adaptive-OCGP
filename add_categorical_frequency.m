function x=add_categorical_frequency(T)
    x = [];

    [genre, ~, index] = unique(T.EnzymeClassification);
    tbl = tabulate(T.EnzymeClassification);
    freq = [tbl{:,2}]';
    EnzymeClassification_ = freq(index(:));
    EnzymeClassification = table(EnzymeClassification_);

    [genre, ~, index] = unique(T.PESTRegion);
    tbl = tabulate(T.PESTRegion);
    freq = [tbl{:,2}]';
    PESTRegion_ = freq(index(:));
    PESTRegion = table(PESTRegion_);

    [genre, ~, index] = unique(T.SignalPeptide);
    tbl = tabulate(T.SignalPeptide);
    freq = [tbl{:,2}]';
    SignalPeptide = freq(index(:));
    SignalPeptide_t = table(SignalPeptide);
    
    [genre, ~, index] = unique(T.Essentiality);
    tbl = tabulate(T.Essentiality);
    freq = [tbl{:,2}]';
    Essentiality_ = freq(index(:));
    Essentiality = table(Essentiality_);
    
    [genre, ~, index] = unique(T.Localization);
    tbl = tabulate(T.Localization);
    freq = [tbl{:,2}]';
    Localization_ = freq(index(:));
    Localization = table(Localization_);
    
    categoricalFeatures = [EnzymeClassification SignalPeptide_t Localization];
   
    for i=1:size(categoricalFeatures,2)
        x = [x,table2array(categoricalFeatures(:,i))];
    end
end