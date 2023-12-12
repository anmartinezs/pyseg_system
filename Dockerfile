# Copyright 2018 Antonio Martinez-Sanchez "an.martinez.s.sw@gmail.com" and
# Vladan Lucic "vladan@biochem.mpg.de" (pyto package)
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

FROM ubuntu:22.04

WORKDIR /usr/local/pyseg_system
COPY . .
RUN rm -rf sys/soft

RUN cd sys && chmod u+x *.sh && ./install_ubuntu_apt_pkgs_docker.sh && ./install_third_parties.sh  \
    && ./install_miniconda.sh && ./install_conda_pkgs.sh
RUN cd sys && ./set_bashrc_env.sh
RUN chmod u+x *.sh && ./clean_out_data.sh


