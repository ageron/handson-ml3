
We put new files under the `constant/` subdirectory.
They are copyright O1 Software Network, and are MIT licensed.

Feel free to put source you're not yet sharing
under `private/`, or a directory named for you, e.g. `sweeper/`.
Ensure that .gitignore will prevent commits of such files --
verify that `git status` remains clean.

----

# big files

Git is not a good fit for "large" files,
especially if they change often, or are binary.
If you add a 1 GiB .mp4 video to the repo,
then every team member's clone will have to download that 1 GiB.
If you delete it in a subsequent commit, it just moves the
giant file to the `.git/` history, which is still a problem.

The `constant/out/` subdirectory is a good place to put large files,
since it is ignored by git.

Notebook .ipynb files can be large, so consider trimming
image results with `nbstripout` before committing.
On the flip side, you might want GitHub to display
analysis images, so it's a judgement call, just
understand the tradeoffs and make a conscious decision.

The `.pre-commit-config.yaml` hooks will remind you
if you try to commit a file that is "large".
It triggers at 100 KiB, which most source code will fit within.
Feel free to override this if you really intend to commit a big binary.

----

# features

Start by creating an issue at
https://github.com/O1SoftwareNetwork/handson-ml3/issues ,
and assign it to yourself.

Then create a branch named for the issue.
When the feature is complete,
ensure that `make lint` is clean,
create a pull request,
assign it to someone else for review,
and merge the approved PR down to `main`.
We use the "squash and merge" option
to keep the commit history tidy.
Now you should delete that feature branch.
Create a new issue if there's still more to do.
It should be rare that you have more than
one or two branches at a time.

`git fetch --prune` will remove any stale local tracking branches.

Read and critique your own code before assigning any review tasks.
Delete commented code, tidy up TODOs, try to make
it easy for the reviewer to understand your proposed changes.
Small PRs are easy to quickly review and approve.
Extract helper functions where you see copy-n-paste repetition.
Adding unittests will often be a good way to demonstrate
how you expect the code to be used.
Rename obscure variables, add a docstring to any function
whose name does not make it self-explanatory.
Consider adding optional type annotations to function signatures.

If you're producing any scatter plots,
verify that both axes are labeled.
Using `import seaborn as sns` can be a very convenient way
to have that happen automatically.

----

# reviewing

Why do we review before a merge-to-`main`?

For code that will run in production,
a review verifies that other team members
will be able to understand and maintain
that code down the road.

For all code (or term papers!), having a
second set of eyes  look over it will often catch
issues that slipped past  the original author.
It offers an excellent teaching opportunity
to share best practices and handy libraries
that the author might not have known about.

And the author is also teaching team members about
the new code, and the techniques and libraries
that it uses.

## one-day rule

Reviews take a day, at most. Period.
They are not onerous, they are not an obstacle,
they do not get in a developer's way.

Reviews are here to help you.
We all learn from them.

Reviewers should respond within 24 hours of a review request.
Send a "thumbs up LGTM" click, or a review,
or a comment that is the beginning of a review,
or at least a "need to consider this, will get back to you tomorrow."
If reviewers are silent for 24 hours,
they have waived the right to improve this particular commit,
and the PR is auto-approved.
Author may merge it to main at their leisure.

## author performs merge

Who should merge to `main`?

The project owner (Jim), or any reviewer, _could_ click "squash merge".
But good manners suggests we should wait for the original author
to merge the PR.

Why? Time has passed, review remarks have come rolling in,
new ideas have suggested themselves.
It is common for an author to wish to make some minor
last-minute edits prior to final commit and merge.
Being small, or in line with reviewer suggestions,
these are not subject to awaiting another round of review.

----

# rebasing

We don't rebase.

Never use `--force` in a `git push` command.

Using `--amend` is fine, as long as you've
not yet pushed that commit to GitHub.

As mentioned above,
use the `--squash` option  when merging a pull request.
The GitHub web UI makes this very easy.
