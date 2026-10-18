# Reference LAPACK with LAPACKE

The `lapack-reference` overlay is copied from microsoft/vcpkg at
`bf19794383e8fb95eba5aa553235bc8ec2271688`, under the included MIT license.
The pinned upstream port does not expose LAPACKE. This overlay adds a `lapacke`
feature and fixes up its installed CMake package alongside the LAPACK package.
All source versions, checksums, patches, and other build behavior are retained.
