%One-hot encoding for categorical features

function x=add_categorical_onehot(T,category_add)
    x = [];

    [genre, ~, index] = unique(T.EnzymeClassification);
    mat = logical(accumarray([(1:numel(index)).' index], 1));
    Hydrolases = mat(:,1);
    Lyases = mat(:,2);
    NotEnzyme = mat(:,3);
    Oxireductases = mat(:,4);
    Transferases = mat(:,5);
    Translocases = mat(:,6);
    EnzymeClassification = table(Hydrolases,Lyases,NotEnzyme,Oxireductases,Transferases,Translocases);

    [genre, ~, index] = unique(T.PESTRegion);
    mat = logical(accumarray([(1:numel(index)).' index], 1));
    Poor = mat(:,1);
    Potential = mat(:,2);
    PESTRegion = table(Poor,Potential);

    [genre, ~, index] = unique(T.SignalPeptide);
    SignalPeptide = index-1;
    SignalPeptide_t = table(index);
    
    
    [genre, ~, index] = unique(T.Essentiality);
    mat = logical(accumarray([(1:numel(index)).' index], 1));
    Essential = mat(:,1);
    NonEssential = mat(:,2);
    UK = mat(:,3);
    Essentiality = table(Essential,NonEssential,UK);

    
    [genre, ~, index] = unique(T.Localization);
    mat = logical(accumarray([(1:numel(index)).' index], 1));
    Chloroplast = mat(:,1);
    Cytoplasmic = mat(:,2);
    Extracellular = mat(:,3);
    Lysosomal = mat(:,4);
    Mitochondrial = mat(:,5);
    Nuclear = mat(:,6);
    PlasmaMembrane = mat(:,7);
    Localization = table(Chloroplast,Cytoplasmic,Extracellular,Lysosomal,Mitochondrial,Nuclear,PlasmaMembrane);
    
    [genre, ~, index] = unique(T.TransmembraneHelices);
    TransmembraneHelices_ = rescale(index);
    TransmembraneHelices = table(TransmembraneHelices_);
   
    categoricalFeatures = [Essentiality PESTRegion];
    
    for i=1:size(categoricalFeatures,2)
        x = [x,table2array(categoricalFeatures(:,i))];
    end
end