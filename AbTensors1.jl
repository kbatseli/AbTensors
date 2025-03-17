### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ d1344690-8c34-4b78-bf28-80740ceb12dd
using LinearAlgebra, PlutoUI, Distributions

# ╔═╡ e2839e3e-b667-41a9-8763-88d39ee24254
html"""
<style>
pluto-helpbox { 
		display: none; 
}
	main{
		margin: 0 auto;
		max-width: 20000px;
		padding-left: max(160px, 10%);
		padding-right: max(160px, 10%);
	}
</style>
"""

# ╔═╡ ea45707a-2c33-42d2-bc18-6676223b2576
md"""
This notebook is part of the experiments section of the article [_"Constructing structured tensor priors for Bayesian inverse problems", Kim Batselier._](https://arxiv.org/abs/2406.17597) and is licensed under the [MIT OSS license](https://www.tldrlegal.com/license/mit-license?ref=fossa.com#fulltext).

### Sampling priors of $(A,b)$-constrained tensors
---
A tensor $\mathcal W \in \mathbb{R}^{J_1 \times J_2 \times \cdots \times J_D}$ is an $(A,b)$-constrained tensor when there is a matrix $A \in \mathbb{R}^{I \times J_1\cdots J_D}$ and vector $b \in \mathbb{R}^I$ such that ${A} \; \mathrm vec(\mathcal W) = b.$

The class of $(\mathcal A,b)$-constrained tensors contains a wide variety of structured tensors. The purpose of this notebook is to demonstrate this variety by using **Theorems 3.1, 4.5 and 5.1** to construct and sample different priors.
"""

# ╔═╡ f957bab7-9ab8-49e8-a882-2739ee605745
md"""
##### 1. Tensors with known entries: lower triangular tensors
Tensors with known entries are quite common. The known entry is usually zero, giving rise to particular banded or triangular structures. Let's sample the Gaussian prior of a lower triangular tensor. From Definition 2.1 we know that a lower triangular tensor $\mathcal W$ has zero entries $w_{j_1,\ldots,j_D}$ when for some consecutive index pair $j_d,j_{d+1}$ we have that $j_d-j_{d+1}< 0$. We can define a $J(J-1)/2 \times J^2$ matrix $S$ that has a single nonzero element per row such that $s_{i,\overline{j_1j_2}} = 1$ when $j_1-j_2<0$. Then the $A$ matrix that describes a lower triangular tensor is

$A = \begin{pmatrix}A_1 \\ A_2 \\ \vdots \\ A_{D-1} \end{pmatrix} = \begin{pmatrix}
        {S}  \otimes  {I}_J  \otimes   \cdots  \otimes  {I}_J\\
        {I}_J \otimes  {S} \otimes  \cdots \otimes  {I}_J\\
        \vdots \\
        {I}_J \otimes  {I}_J \otimes  \cdots \otimes  {S}
\end{pmatrix} \in \mathbb{R}^{\frac{(D-1)(J-1)J^{D-1}}{2} \times J^D}.$

The right nullspace of $A$ can be computed recursively through **Algorithm 3.1** without ever explicitly constructing $A$.
"""

# ╔═╡ 5ba2db0f-7b3c-48b0-8fb2-b8e363b92f65
begin
	order_slider = @bind order PlutoUI.Slider(2:5,default=2)
	dim_slider = @bind dim PlutoUI.Slider(2:6,default=2)
	
	md"""
	 Playing around with the order and dimension sliders below will automatically generate a new tensor sample.
	
	Order $D$ of the tensor: $(order_slider)
	
	Dimension $J$ of the tensor: $(dim_slider)
	"""
end

# ╔═╡ b441b267-a417-4a8a-bcb5-abf54fee328d
md"""
Sampling a lower triangular tensor of order $(order) and dimension $(dim):
"""

# ╔═╡ 07320fc9-3d83-436d-8bc1-5e18599bfa5c
begin
	# construct S
	S=zeros(Int64(dim*(dim-1)/2),dim^2)
	rowcounter=1
	for i=1:dim-1
		for j=i+1:dim
			S[rowcounter,i+(j-1)*dim]=1
			rowcounter+=1
		end
	end
	# basis nullspace through Algorithm 3.1 
	V2 = nullspace(S)
	for j=1:order-2
		V2= kron(V2,Matrix(1.0I,dim,dim))
	end
	for k=2:order-1
		Ak = Matrix(1.0I,dim,dim)
		for j=2:order-1
			if j==k
				Ak = kron(Ak,S)
			else
				Ak = kron(Ak,Matrix(1.0I,dim,dim))
			end
		end
		V3 = nullspace(Ak*V2)
		V2 = V2*V3
	end
	V2[abs.(V2).< 10.0^(-10)].=0.0
	reshape(V2*randn(size(V2,2),1), ntuple(i -> dim, order) )
end

# ╔═╡ 9bcf944e-746b-4881-a022-fe947d76a7f4
md"""
##### 2. Tensors with known sum of entries
Tensors $\mathcal W$ with a known sum of entries are also $(A,b)$-constrained. In this example we sample tensors for which the sum over the last index always adds up to a value of 1: 

$\forall j_1, j_2, \ldots, j_{D-1}: \sum_{j_D} w_{j_1, j_2, \ldots, j_D} = b_{j_1, j_2, \ldots, j_{D-1}}=1.$  
From **Lemma 2.3** we know that in this case $A = 1_J^T \otimes I_J \otimes \cdots \otimes I_J$. It is straightforward to verify that a basis for the right nullspace of $A$ is

$\begin{pmatrix}  1 & 1 & \cdots & 1 \\ -1 & 0 & \cdots & 0 \\ 0 & -1 & \cdots &0 \\ 0 & 0 & \cdots & -1\end{pmatrix}\otimes I_J \otimes \cdots \otimes I_J.$

Sampling the prior can now be done without every constructing a basis for the nullspace explicitly since

$\sqrt{P_0}\; x = \left( \begin{pmatrix}  1 & 1 & \cdots & 1 \\ -1 & 0 & \cdots & 0 \\ 0 & -1 & \cdots &0 \\ 0 & 0 & \cdots & -1\end{pmatrix}\otimes I_J \otimes \cdots \otimes I_J \right) \;x = \begin{pmatrix}  I_{J^{D-1}} & I_{J^{D-1}} & \cdots & I_{J^{D-1}} \\ -I_{J^{D-1}} & 0 & \cdots & 0 \\ 0 & -I_{J^{D-1}} & \cdots &0 \\ 0 & 0 & \cdots & -I_{J^{D-1}}\end{pmatrix} \;x$

$=\begin{pmatrix} I_{J^{D-1}} & I_{J^{D-1}} & \cdots & I_{J^{D-1}} \\ -I_{J^{D-1}} & 0 & \cdots & 0 \\ 0 & -I_{J^{D-1}} & \cdots &0 \\ 0 & 0 & \cdots & -I_{J^{D-1}}\end{pmatrix} \; \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_{J-1} \end{pmatrix} = \begin{pmatrix} x_1 + x_2 + \cdots + x_{J-1} \\ -x_1 \\ -x_2 \\ \vdots \\ -x_{J-1}\end{pmatrix}.$

It is therefore sufficient to sample $x$ from a standard normal distribution and do the operations on the partitions of $x$ as described above to generate the desired sample.
"""

# ╔═╡ 4e9e54f9-1680-4849-bcfd-177d03d75824

begin
	order2_slider = @bind D PlutoUI.Slider(2:5,default=2)
	dim2_slider = @bind N PlutoUI.Slider(5:10,default=5)
	
	md"""
	Order $D$ of the tensor: $(order2_slider)
	
	Dimension $J$ of the tensor: $(dim2_slider)
	"""
end

# ╔═╡ dcb8cf30-5937-41d4-bfda-f6354189c59b
begin	
	W0=ones(N^D)/N
	x = randn(N^(D-1),(N-1))
	s = zeros(N^D)
	s[1:N^(D-1)] = sum(x,dims=2)
	for i in (1:N-1)
		s[i*N^(D-1)+1:(i+1)*N^(D-1)] = -x[:,i] 
	end
	sample = reshape(W0 + s, ntuple(i -> N,D) )
	summed = reshape(sum(reshape(sample, (N^(D-1),N)),dims=2),ntuple(i -> N, D-1))
	sample
	#sum(reshape(W,[N^(D-1),N]),2)
end

# ╔═╡ 98aa6e6e-6214-430c-b857-58ddb926eaf2
md"""
And the tensor we obtain by summing over the last index is:
"""

# ╔═╡ 4da877a9-12f9-4367-9ad9-dbb4eb677507
summed

# ╔═╡ c1d6d89f-5106-438a-b943-d7969d69bcfa
md"""
##### 3. $P$-invariant tensors
Tensors $\mathcal W$ whose vectorization is an eigenvector of a permutation matrix $P$ with eigenvalue 1 are $(A,b)$-constrained tensors with by $A=I-P$ and $b=0$. These kind of tensors have the property of being invariant under the permutation $P$, since $P\, w=w$. For a given permutation matrix $P$ and its order $K$, one can use **Theorems 4.5** and **5.1** to construct the covariance matrix of the prior.  
"""

# ╔═╡ 4af8fc07-ab13-4888-a9c3-caef5f27c297
begin
 	order3_slider = @bind Dsym PlutoUI.Slider(2:3,default=2)
	dim3_slider = @bind Nsym PlutoUI.Slider(5:10,default=5)
	md"""
	###### 3.1 Symmetric tensors
	Symmetric tensors $\mathcal W$ are tensors for which entries are invariant under any index permutation. The permutation matrix $P$ in the symmetric case consists of cyclic permutations where each each cycle is the entry $w_{j_1, \ldots,j_D}$ and all corresponding index permutations $w_{\pi (j_1,\ldots, j_D)}$. For example, in the case $D=2$ and $J=2$ the permutation matrix $P$ consists of $3$ cyclic permutations

	$\begin{align*}
	w_{1,1} \mapsto w_{1,1}\\
	 w_{2,1} \mapsto w_{1,2} \\
	w_{1,2} \mapsto w_{2,1} \\
	w_{2,2} \mapsto w_{2,2}.
	\end{align*}$  

	The order of $P$ in this case is 2 since $P^2=I$. According to **Theorem 4.5** we then have that the square root of the covariance matrix is

	$\sqrt{P_0} = \frac {P + P^2}{2}.$
	
	When $D=3$, the order of the corresponding permutation matrix is $6$ and hence
	
	$\sqrt{P_0} = \frac {P + P^2 + P^3 + P^4 + P^5 + P^6}{6}.$

	Sampling from these priors can then be done via **Algorithm 4.1** where a standard normal sample $x$ is generated and permuted $K$ times:

	Order $D$ of the tensor: $(order3_slider)
	
	Dimension $J$ of the tensor: $(dim3_slider)
	"""
end

# ╔═╡ f4e2900d-8533-4f33-b66d-37d6873a91dc
begin	
	if Dsym ==2
		x2=randn(Nsym,Nsym)
		x2sym = (x2+x2')/2
	else
		x3=randn(Nsym,Nsym,Nsym)
		x3sym= ( x3 + permutedims(x3,(1,3,2)) + permutedims(x3,(2,1,3)) + permutedims(x3,(2,3,1)) + permutedims(x3,(3,1,2)) + permutedims(x3,(3,2,1))) / 6
	end	
end

# ╔═╡ b87204f4-6222-44d2-9a78-4b730eae34c3
begin
 	order4_slider = @bind Dhankel PlutoUI.Slider(2:4,default=2)
	dim4_slider = @bind Nhankel PlutoUI.Slider(3:10,default=3)
	md"""
	###### 3.2 Hankel tensors
	Hankel tensors $\mathcal W$ are tensors for which entries with a constant index sum $j_1+\cdots+j_D$ have the same numerical value. The order $K$ of the corresponding permutation matrix $P$ grows very quickly. For example, when $D=2$ and $J=20$ the order $K$ is the least common multiple of $1,2,\ldots,20 = 232,792,560$. **Theorem 5.1**, however, allows us to construct a matrix $\sqrt{P_0} \in \mathbb{R}^{J^D \times R}$, where $R$ is the number of permutation cycles. For Hankel tensors we have that $R = D(J-1)+1$.

	Order $D$ of the tensor: $(order4_slider)
	
	Dimension $J$ of the tensor: $(dim4_slider)
	"""
end

# ╔═╡ 8e8df2c8-50c3-4b22-be5c-cf766bb94492
begin
	if Dhankel==2
		V=zeros(Nhankel^2,Dhankel*(Nhankel-1)+1)
		for j1 = 1 : Nhankel
			for j2 = 1 : Nhankel
				V[j1+(j2-1)*Nhankel,j1+j2-(Dhankel-1)]=1; 
			end
		end
		reshape(V*randn(Dhankel*(Nhankel-1)+1,1),(Nhankel,Nhankel))
	elseif Dhankel==3
		V=zeros(Nhankel^3,Dhankel*(Nhankel-1)+1)
		for j1 = 1 : Nhankel
			for j2 = 1 : Nhankel
				for j3 = 1 : Nhankel
					V[j1+(j2-1)*Nhankel+(j3-1)*Nhankel^2,j1+j2+j3-(Dhankel-1)]=1; 
				end
			end
		end
		reshape(V*randn(Dhankel*(Nhankel-1)+1,1),(Nhankel,Nhankel,Nhankel))		
	else
		V=zeros(Nhankel^4,Dhankel*(Nhankel-1)+1)
		for j1 = 1 : Nhankel
			for j2 = 1 : Nhankel
				for j3 = 1 : Nhankel
					for j4 = 1 : Nhankel
						V[j1+(j2-1)*Nhankel+(j3-1)*Nhankel^2+(j4-1)*Nhankel^3,j1+j2+j3+j4-(Dhankel-1)]=1; 
					end
				end
			end
		end
		reshape(V*randn(Dhankel*(Nhankel-1)+1,1),(Nhankel,Nhankel,Nhankel,Nhankel))	
	end
end

# ╔═╡ c3a37ff4-cc6f-4977-bd37-00b7b824937c
md"""
---
This notebook is written as part of the project Sustainable learning for Artificial Intelligence from noisy large-scale data (with project number VI.Vidi.213.017) which is financed by the Dutch Research Council (NWO).

Technische Universiteit Delft hereby disclaims all copyright interest in this notebook written by Kim Batselier.

Fred van Keulen, Dean of Mechanical Engineering

Copyright <2024> <Kim Batselier, VI.Vidi.213.017>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Distributions = "~0.25.107"
PlutoUI = "~0.7.58"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "8b9b357c208ba43032c96598bc4bdf1183ddb210"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0f748c81756f2e5e6854298f11ad8b2dfae6911a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "c955881e3c981181362ae4088b35995446298b80"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.14.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "0f4b5d62a88d8f59003e43c25a8a90de9eb76317"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.18"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7c302d7a5fec5214eb8a5a4c466dcf7a51fcf169"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.107"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "bfe82a708416cf00b73a3198db0859c82f741558"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.10.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "71a22244e352aa8c5f0f2adde4150f62368a3f2e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.58"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─d1344690-8c34-4b78-bf28-80740ceb12dd
# ╟─e2839e3e-b667-41a9-8763-88d39ee24254
# ╟─ea45707a-2c33-42d2-bc18-6676223b2576
# ╟─f957bab7-9ab8-49e8-a882-2739ee605745
# ╟─5ba2db0f-7b3c-48b0-8fb2-b8e363b92f65
# ╟─b441b267-a417-4a8a-bcb5-abf54fee328d
# ╟─07320fc9-3d83-436d-8bc1-5e18599bfa5c
# ╟─9bcf944e-746b-4881-a022-fe947d76a7f4
# ╟─4e9e54f9-1680-4849-bcfd-177d03d75824
# ╟─dcb8cf30-5937-41d4-bfda-f6354189c59b
# ╟─98aa6e6e-6214-430c-b857-58ddb926eaf2
# ╟─4da877a9-12f9-4367-9ad9-dbb4eb677507
# ╟─c1d6d89f-5106-438a-b943-d7969d69bcfa
# ╟─4af8fc07-ab13-4888-a9c3-caef5f27c297
# ╟─f4e2900d-8533-4f33-b66d-37d6873a91dc
# ╟─b87204f4-6222-44d2-9a78-4b730eae34c3
# ╟─8e8df2c8-50c3-4b22-be5c-cf766bb94492
# ╟─c3a37ff4-cc6f-4977-bd37-00b7b824937c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
