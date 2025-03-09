import { type CollectionEntry, getCollection } from "astro:content";

/** filter out draft posts based on the environment */
export async function getAllPosts(): Promise<CollectionEntry<"post">[]> {
	return await getCollection("post", ({ data }) => {
		return import.meta.env.PROD ? !data.draft : true;
	});
}

/** get all notes */
export async function getAllNotes(): Promise<CollectionEntry<"note">[]> {
	return await getCollection("note");
}

/** groups posts by year (based on option siteConfig.sortPostsByUpdatedDate), using the year as the key
 *  Note: This function doesn't filter draft posts, pass it the result of getAllPosts above to do so.
 */
export function groupPostsByYear(posts: CollectionEntry<"post">[]) {
	return posts.reduce<Record<string, CollectionEntry<"post">[]>>((acc, post) => {
		const year = post.data.publishDate.getFullYear();
		if (!acc[year]) {
			acc[year] = [];
		}
		acc[year]?.push(post);
		return acc;
	}, {});
}

/** groups notes by year, using the year as the key */
export function groupNotesByYear(notes: CollectionEntry<"note">[]) {
	return notes.reduce<Record<string, CollectionEntry<"note">[]>>((acc, note) => {
		const year = note.data.publishDate.getFullYear();
		if (!acc[year]) {
			acc[year] = [];
		}
		acc[year]?.push(note);
		return acc;
	}, {});
}

/** 返回所有文章的标签（包含重复标签） */
export function getAllPostTags(posts: CollectionEntry<"post">[]) {
	return posts.flatMap((post) => [...post.data.tags]);
}

/** 返回所有笔记的标签（包含重复标签） */
export function getAllNoteTags(notes: CollectionEntry<"note">[]) {
	return notes.flatMap((note) => [...(note.data.tags || [])]);
}

/** 返回所有文章和笔记的标签（包含重复标签） */
export function getAllContentTags(posts: CollectionEntry<"post">[], notes: CollectionEntry<"note">[]) {
	return [...getAllPostTags(posts), ...getAllNoteTags(notes)];
}

/** 返回所有文章和笔记的唯一标签 */
export function getUniqueContentTags(posts: CollectionEntry<"post">[], notes: CollectionEntry<"note">[]) {
	return [...new Set(getAllContentTags(posts, notes))];
}

/** 返回每个唯一标签的计数 - [[tagName, count], ...] */
export function getUniqueContentTagsWithCount(posts: CollectionEntry<"post">[], notes: CollectionEntry<"note">[]): [string, number][] {
	return [
		...getAllContentTags(posts, notes).reduce(
			(acc, t) => acc.set(t, (acc.get(t) ?? 0) + 1),
			new Map<string, number>(),
		),
	].sort((a, b) => b[1] - a[1]);
}

/** returns all tags created from posts (inc duplicate tags)
 *  Note: This function doesn't filter draft posts, pass it the result of getAllPosts above to do so.
 *  */
export function getAllTags(posts: CollectionEntry<"post">[]) {
	return posts.flatMap((post) => [...post.data.tags]);
}

/** returns all unique tags created from posts
 *  Note: This function doesn't filter draft posts, pass it the result of getAllPosts above to do so.
 *  */
export function getUniqueTags(posts: CollectionEntry<"post">[]) {
	return [...new Set(getAllTags(posts))];
}

/** returns a count of each unique tag - [[tagName, count], ...]
 *  Note: This function doesn't filter draft posts, pass it the result of getAllPosts above to do so.
 *  */
export function getUniqueTagsWithCount(posts: CollectionEntry<"post">[]): [string, number][] {
	return [
		...getAllTags(posts).reduce(
			(acc, t) => acc.set(t, (acc.get(t) ?? 0) + 1),
			new Map<string, number>(),
		),
	].sort((a, b) => b[1] - a[1]);
}
