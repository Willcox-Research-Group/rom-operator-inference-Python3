(sec-normalization)=
# Data Normalization

:::{attention}
This page is under construction.
:::

- Here are a few common things to do in order to preprocess the data.
- There are other interesting things you could do [Boris rotating OpInf paper] but here we just demonstrate the package tools.

## Shifting / Centering

- What low-dimensional approximation to use for the state is a modeling choice. Often that includes centering the data before computing the principal components (refer to PCA).
- Compute the center with thingy or using one of classes in the next section.
- Reference Renee's thesis/paper for building BC's into a model.

## Scaling

- If we have multiple variables that have super different scales, then we need to do some preprocessing first in order for one variable to not overwhelm the other (this is a common machine learning thing, refer to scikit-learn docs).
- This is not the same as preconditioning the least-squares problem.

## The SnapshotTransformer Class

- Table with available `SnapshotTransformer` transformations.
- Demonstrate `SnapshotTransformer`. Note that it can do centering first as well.

## Multivariable Transformers

- Demonstrate `SnapshotTransformerMulti`. Note that it can do centering first as well.
